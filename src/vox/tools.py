"""Tool definitions and intent detection for concurrent execution."""

from __future__ import annotations

import datetime
import logging
import re
import sys
import time as _time
from dataclasses import dataclass

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Content-routing keywords — triggers unrestricted model for mature content.
# Full keyword list loaded from persona/content_keywords.txt (gitignored).
# Committed code only has PG defaults; local config extends them.
# ---------------------------------------------------------------------------
def _load_content_keywords() -> re.Pattern:
    from pathlib import Path
    words = [
        "naked", "nude", "nsfw", "topless", "lingerie", "underwear",
        "bikini", "sexy", "seductive", "erotic", "sensual", "provocative",
        "undress", "strip", "bare", "exposed", "adult", "mature", "explicit",
    ]
    kw_file = Path(__file__).parent.parent.parent / "persona" / "content_keywords.txt"
    if kw_file.exists():
        custom = [w.strip() for w in kw_file.read_text().splitlines()
                  if w.strip() and not w.startswith("#")]
        if custom:
            words = list(dict.fromkeys(words + custom))  # merge, dedupe, preserve order
    return re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", re.IGNORECASE)

_NSFW_KEYWORDS = _load_content_keywords()


def _load_content_mappings() -> dict:
    """Load NSFW/pose scene descriptor mappings from external JSON (gitignored)."""
    import json
    from pathlib import Path
    mappings_file = Path(__file__).parent.parent.parent / "persona" / "content_mappings.json"
    if mappings_file.exists():
        try:
            return json.loads(mappings_file.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("Failed to load content_mappings.json: %s", e)
    return {}

_CONTENT_MAPPINGS = _load_content_mappings()


# ---------------------------------------------------------------------------
# Intent detection — fast keyword matching, no LLM needed (~1ms)
# ---------------------------------------------------------------------------

@dataclass
class DetectedIntent:
    """A detected tool intent with the function to call."""

    tool_name: str
    args: dict
    bridge_phrase: str  # what VOX says while the tool runs


# Pattern → (tool_name, arg_builder, bridge_phrase)
# arg_builder receives (match, full_text) so it can extract args from the full user message
_INTENT_PATTERNS: list[tuple[re.Pattern, str, callable, str]] = []

# Track the last tool and its args so follow-up requests ("another one") can repeat
_last_tool: str | None = None
_last_tool_args: dict = {}

# Image generation progress callback — set by web.py to push step updates
# Signature: fn(step: int, total_steps: int) -> None
_image_progress_fn: callable | None = None

# Image saved callback — set by web.py to push image inline immediately
# Signature: fn(filename: str) -> None
_image_saved_fn: callable | None = None

# Rate limit — minimum seconds between image generation requests
_last_image_gen_time: float = 0.0
_IMAGE_COOLDOWN_SEC = 5.0

# Follow-up patterns — short messages that mean "do that again"
_FOLLOWUP_PATTERN = re.compile(
    r"^(another\s+one|one\s+more|do\s+(it|that)\s+again|again|more|another)"
    r"$",
    re.IGNORECASE,
)


def _add_pattern(pattern: str, tool_name: str, arg_builder: callable, bridge: str):
    _INTENT_PATTERNS.append((re.compile(pattern, re.IGNORECASE), tool_name, arg_builder, bridge))


def _extract_email(text: str) -> str:
    """Extract an email address from text, try contacts, fall back to USER_EMAIL."""
    m = re.search(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", text)
    if m:
        return m.group(0)
    # Try resolving a name via contacts
    name_match = re.search(
        r"\b(?:email|mail|send)\s+(?:to\s+)?(\w+(?:\s+\w+)?)\b",
        text, re.IGNORECASE,
    )
    if name_match:
        name = name_match.group(1).strip()
        # Skip generic words that aren't names
        if name.lower() not in ("me", "it", "this", "that", "the", "them", "her", "him"):
            try:
                from vox.contacts import resolve_email
                emails = resolve_email(name)
                if emails:
                    return emails[0]
            except Exception:
                log.debug("Contact lookup failed for %s", name)
    from vox.config import USER_EMAIL
    return USER_EMAIL


def _extract_url(text: str) -> str:
    """Extract a URL from text."""
    m = re.search(r"https?://\S+", text)
    return m.group(0).rstrip(".,;!?)") if m else ""


def _extract_location(text: str) -> str:
    """Extract a location/address from text for map lookups."""
    # Strip the command part, keep the location
    cleaned = re.sub(
        r"\b(show|get|pull\s*up|find|give)\s+(me\s+)?(a\s+)?"
        r"(satellite|aerial|map|street\s*view|bird.?s?\s*eye|top\s*down)\s*"
        r"(view|image|photo|picture|map)?\s*(of|for)?\s*",
        "", text, flags=re.IGNORECASE,
    ).strip()
    # Also strip leading articles and prepositions
    cleaned = re.sub(r"^(the|a|an|at|of|for|to|in)\s+", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned if cleaned else text


def _is_selfie_request(text: str) -> bool:
    """Check if the text is asking for a selfie / picture of the assistant."""
    return bool(re.search(
        r"\b(selfie|selfy)\b"
        r"|\b(picture|pic|photo|image)\s+(of\s+)?(you|yourself|your\s*(self|face|body))"
        r"|\b(what\s+do\s+you\s+look\s+like)"
        r"|\b(show\s+(me\s+)?(yourself|what\s+you\s+look\s+like))"
        r"|\b(take\s+a\s+(pic|picture|photo|selfie|snap|shot))\b"
        r"|\b(send|email|mail|give)\s+me\s+a\s+(selfie|pic|picture|photo)"
        r"|\b(let\s+me\s+see\s+you)\b",
        text, re.IGNORECASE,
    ))


def _build_persona_prompt(text: str) -> str:
    """Build an SD prompt using the persona description + any scene context from the user."""
    from vox.config import VOX_PERSONA_DESCRIPTION, VOX_PERSONA_STYLE

    # Extract scene/context modifiers from the user's request.
    # Strategy: strip everything that ISN'T a scene/pose/location modifier.
    scene = text.strip()

    # Detect easter egg trigger word BEFORE stripping it (gates NSFW behavior)
    _nsfw_unlocked = bool(re.search(r"\bohwx\s+\w+\b", scene, re.IGNORECASE))
    # Strip LoRA trigger words the user may have typed (we inject them ourselves)
    scene = re.sub(r"\bohwx\s+\w+\b", "", scene, flags=re.IGNORECASE).strip()
    # Strip connectors that survive after stripping: "with your", "and your", "you're"
    scene = re.sub(r"\b(with\s+(your|you)|and\s+your|you'?re)\b", "", scene, flags=re.IGNORECASE)

    # Pre-process: convert action phrases into SD-friendly descriptors
    # Mappings loaded from persona/content_mappings.json (gitignored)
    if _nsfw_unlocked:
        for mapping in _CONTENT_MAPPINGS.get("nsfw_unlocked", []):
            scene = re.sub(mapping["pattern"], mapping["replacement"], scene, flags=re.IGNORECASE)
        # Clean "nude and topless" → "nude, topless" (SD prefers comma-separated)
        scene = re.sub(r"\b(nude|topless|bottomless)\s+and\s+(nude|topless|bottomless)\b",
                       lambda m: f"{m.group(1)}, {m.group(2)}", scene, flags=re.IGNORECASE)
    else:
        # Without easter egg: strip NSFW terms entirely so prompt stays SFW
        strip_pat = _CONTENT_MAPPINGS.get("nsfw_locked_strip", "")
        if strip_pat:
            scene = re.sub(strip_pat, "", scene, flags=re.IGNORECASE)

    # "and show me a selfie" / "another selfie" → just remove, we already know it's a selfie
    scene = re.sub(
        r"\b(and\s+)?(show|send|give|take)\s+(me\s+)?(a\s+)?(selfie|pic|picture|photo)\b",
        "", scene, flags=re.IGNORECASE,
    )
    scene = re.sub(
        r"\b(another|one\s+more|a\s+new)\s+(selfie|pic|picture|photo)\b",
        "", scene, flags=re.IGNORECASE,
    )
    # "surprise me with something" → extract the adjective that follows
    scene = re.sub(r"\bsurprise\s+me\s+with\s+(something\s+)?", "", scene, flags=re.IGNORECASE)

    # Convert natural-language actions to SD-friendly pose descriptors
    # General pose mappings (SFW-safe) from external config
    for mapping in _CONTENT_MAPPINGS.get("pose_mappings", []):
        scene = re.sub(mapping["pattern"], mapping["replacement"], scene, flags=re.IGNORECASE)
    # "laying down/on side/stomach" — dynamic replacement
    scene = re.sub(
        r"\blay(ing)?\s+(down|on\s+(your|my|the)\s+(side|stomach|belly))\b",
        lambda m: f"lying {m.group(2)}", scene, flags=re.IGNORECASE,
    )

    # Strip imperatives / exclamations / conversational commands
    scene = re.sub(
        r"\b(do\s+it|just\s+do\s+it|come\s+on|go\s+ahead|right\s+now|hurry)\b",
        "", scene, flags=re.IGNORECASE,
    )
    scene = re.sub(r"\.{2,}|!{2,}", ",", scene)  # "..." and "!!!" → comma

    # Strip "show me" / "let me see" anywhere (not just at start)
    scene = re.sub(
        r"\b(and\s+)?(show|give|let)\s+me\s+(see\s+)?",
        "", scene, flags=re.IGNORECASE,
    )

    # Strip conversational fluff and compliments
    scene = re.sub(
        r"\b(that'?s|thats)\s+(a\s+)?(little|bit|kinda|pretty|really|so|very)\s+",
        "", scene, flags=re.IGNORECASE,
    )
    scene = re.sub(
        r"\b(now|ok|okay|alright|hey|well)\s+",
        "", scene.strip(), flags=re.IGNORECASE,
    )
    scene = re.sub(
        r"^.*?\b(can\s+(i|you)|could\s+(i|you)|i\s+(want|need|like|love|would))\b",
        "", scene, flags=re.IGNORECASE,
    )
    scene = re.sub(
        r"^.*?\b(have\s+a|get\s+a|see\s+a|give\s+me)\b",
        "", scene, flags=re.IGNORECASE,
    )
    # Strip common prefixes
    scene = re.sub(
        r"^(can you|could you|please|hey vox|vox)\s+",
        "", scene.strip(), flags=re.IGNORECASE,
    )
    # Strip selfie-related command words
    scene = re.sub(
        r"^(send|give|show|take|email|mail)\s+(me\s+)?(a\s+)?",
        "", scene.strip(), flags=re.IGNORECASE,
    )
    scene = re.sub(
        r"\b(selfie|selfy|pic|picture|photo|image|snap|shot|portrait)"
        r"\s*(of\s+)?(you|yourself)?\s*",
        "", scene.strip(), flags=re.IGNORECASE,
    )
    # Strip question phrases and LLM-injected meta-questions
    scene = re.sub(
        r"^(what\s+do\s+you\s+look\s+like)\s*",
        "", scene.strip(), flags=re.IGNORECASE,
    )
    scene = re.sub(
        r"^(show\s+(me\s+)?(yourself|what\s+you\s+look\s+like))\s*",
        "", scene.strip(), flags=re.IGNORECASE,
    )
    scene = re.sub(
        r"^(let\s+me\s+see\s+you)\s*",
        "", scene.strip(), flags=re.IGNORECASE,
    )
    scene = re.sub(
        r"\??\s*what\s+(is|are)\s+(your|my)\s+\w+",
        "", scene, flags=re.IGNORECASE,
    )
    scene = re.sub(
        r"\??\s*what\s+do\s+you\s+look\s+like",
        "", scene, flags=re.IGNORECASE,
    )
    # Strip "of yourself" / "of you" anywhere
    scene = re.sub(
        r"\b(of\s+)?(yourself|you)\b", "", scene, flags=re.IGNORECASE,
    )
    # Strip "image of yourself" residue
    scene = re.sub(
        r"\b(full\s*body|ful\s*body)\s*(image|picture|photo|pic)?\b",
        "full body", scene, flags=re.IGNORECASE,
    )
    # Strip email-related tail
    scene = re.sub(
        r"\b(and\s+)?(email|send|mail)\b.*$",
        "", scene, flags=re.IGNORECASE,
    ).strip()
    scene = re.sub(
        r"\b(at|to)\s+\S+@\S+\.\S+.*$",
        "", scene, flags=re.IGNORECASE,
    ).strip()
    # Strip conversational connectors and cleanup
    scene = re.sub(r"^,?\s*", "", scene)
    scene = re.sub(r"\bwhile\s+", ", ", scene)  # "while squeezing" → ", squeezing"
    scene = re.sub(r"\band\s+,", ",", scene)  # "and ," → ","
    scene = re.sub(r",\s*,+", ",", scene)  # collapse multiple commas
    scene = re.sub(r"\s{2,}", " ", scene)
    # Strip orphan short words left from aggressive stripping
    scene = re.sub(r"\b(a|an|the|just|some|with|and|or|in|on|at|my|your|me)\b",
                   "", scene, flags=re.IGNORECASE)
    scene = re.sub(r",\s*,+", ",", scene)
    scene = re.sub(r"\s{2,}", " ", scene)
    scene = scene.strip().rstrip("?.!,")

    # Build the full prompt: trigger word + persona description + scene + style
    # Prefer persona card data over legacy config vars
    from vox.persona import get_appearance, get_card, get_style_tags
    appearance = get_appearance() or VOX_PERSONA_DESCRIPTION
    style = get_style_tags() or VOX_PERSONA_STYLE

    # Always strip baked-in clothing from appearance — let scene context drive outfit.
    # NSFW scenes need clothing removed; SFW scenes need clothing replaced contextually.
    _clothing_patterns = [
        r"\b(tank\s*top|t-?shirt|shirt|blouse|top|sweater|hoodie|jacket|dress)\b",
        r"\b(sweatpants|pants|jeans|shorts|skirt|leggings|bottoms)\b",
        r"\b(sports?\s*bra|bra|underwear|panties|lingerie)\b",
    ]
    if appearance:
        for pat in _clothing_patterns:
            appearance = re.sub(pat, "", appearance, flags=re.IGNORECASE)
        appearance = re.sub(r",\s*,", ",", appearance)  # collapse double commas
        appearance = re.sub(r",\s*$", "", appearance.strip())

    # Add pose/angle variety for more interesting selfies
    import random
    if _nsfw_unlocked:
        poses = [
            "looking at viewer", "looking over shoulder", "looking up",
            "candid pose", "leaning forward", "sitting", "lying down",
            "standing", "from below angle", "from above angle",
            "close-up face", "medium shot", "three-quarter view",
        ]
        settings = [
            "bedroom", "living room couch", "kitchen", "bathroom mirror",
            "backyard", "car selfie", "bed", "desk", "window light",
        ]
    else:
        # SFW mode: portrait-safe poses and settings only
        # NOTE: "head and shoulders portrait" is injected by the SFW clothing block below,
        # so do NOT include it here to avoid duplicates.
        poses = [
            "looking at viewer", "looking over shoulder", "smiling",
            "candid pose", "standing",
            "close-up face", "three-quarter view",
        ]
        settings = [
            "living room", "kitchen", "backyard", "park",
            "cafe", "office", "window light", "outdoors",
        ]
    # Only add random pose/setting if user didn't specify one
    pose_words = {"sitting", "standing", "lying", "supine", "prone", "kneeling",
                  "leaning", "looking", "close-up", "bent over", "spread",
                  "full body", "face", "angle", "mirror", "selfie", "on back",
                  "on side", "on stomach", "from behind", "from below", "from above"}
    has_pose = scene and any(w in scene.lower() for w in pose_words)
    setting_words = {"bedroom", "kitchen", "bathroom", "couch", "bed", "car",
                     "outside", "backyard", "desk", "window", "shower", "pool",
                     "public", "street", "park", "beach", "office", "gym"}
    has_setting = scene and any(w in scene.lower() for w in setting_words)

    random_additions = []
    if not has_pose:
        random_additions.append(random.choice(poses))
    if not has_setting:
        random_additions.append(random.choice(settings))

    # Inject LoRA trigger word if a LoRA is available
    trigger = ""
    card = get_card()
    if card:
        from vox.lora import get_lora_path, get_trigger_word
        if get_lora_path(card["name"]):
            trigger = get_trigger_word(card["name"])

    # --- Assemble prompt in two halves for SDXL dual CLIP encoders ---
    # prompt_1: subject + scene (≤ 70 tokens for CLIP-L 77-token limit)
    # prompt_2: style/quality tags (for OpenCLIP-G, longer context)
    subject_parts = []
    if trigger:
        subject_parts.append(trigger)
    if appearance:
        subject_parts.append(appearance)
    # SFW mode: enforce clothed portrait — pick outfit from scene context
    if not _nsfw_unlocked:
        scene_lower = (scene or "").lower()
        _pro_words = {"office", "badge", "professional", "business", "corporate",
                      "formal", "work", "meeting", "interview", "headshot"}
        if any(w in scene_lower for w in _pro_words):
            subject_parts.append("clothed, professional attire, blouse, head and shoulders portrait")
        else:
            subject_parts.append("clothed, casual outfit, head and shoulders portrait")
    if scene:
        subject_parts.append(scene)
    if random_additions:
        subject_parts.append(", ".join(random_additions))

    prompt_subject = ", ".join(subject_parts) if subject_parts else "a portrait"

    # Trim subject to ~70 tokens (rough: 1 token ≈ 1 comma-separated phrase or word)
    subject_tokens = [t.strip() for t in prompt_subject.split(",") if t.strip()]
    if len(subject_tokens) > 18:
        subject_tokens = subject_tokens[:18]
        prompt_subject = ", ".join(subject_tokens)

    # Style tags go in prompt_2 via the SDXL dual-encoder split in _generate_image.
    # Append with a clear separator so the splitter can find it.
    if style:
        return f"{prompt_subject}, {style}"
    return prompt_subject


def _extract_image_prompt(text: str) -> str:
    """Extract an image prompt by stripping command words and filler."""
    prompt = re.sub(
        r"^(can you|could you|please|hey vox|vox)\s+",
        "", text.strip(), flags=re.IGNORECASE,
    )
    # Strip "email/mail/send/give/show me" prefix
    prompt = re.sub(
        r"^(email|mail|send|give|show)\s+me\s+",
        "", prompt, flags=re.IGNORECASE,
    )
    # Strip count prefix ("5 pictures of" → "")
    prompt = re.sub(
        r"^\d+\s+",
        "", prompt, flags=re.IGNORECASE,
    )
    # Strip command verbs
    prompt = re.sub(
        r"^(generate|create|draw|make|paint|imagine)\s+",
        "", prompt, flags=re.IGNORECASE,
    )
    # Strip "me" after command verb
    prompt = re.sub(r"^me\s+", "", prompt, flags=re.IGNORECASE)
    # Strip "an image/picture/photo/pics of"
    prompt = re.sub(
        r"^(an?\s+)?(image|picture|photo|artwork|illustration|pic|pics|pictures|photos|images)\s+(of\s+|with\s+)?",
        "", prompt, flags=re.IGNORECASE,
    )
    # Strip email-related tail ("...and email it to foo@bar.com", "...at foo@bar.com")
    prompt = re.sub(r"\b(and\s+)?(email|send)\b.*$", "", prompt, flags=re.IGNORECASE).strip()
    prompt = re.sub(r"\b(at|to)\s+\S+@\S+\.\S+.*$", "", prompt, flags=re.IGNORECASE).strip()
    # Strip purpose tails — "for a ... meme", "for my blog", etc.
    prompt = re.sub(r"\s+for\s+(a|an|my|the|some)\b.*\b(meme|blog|post|project|website|collection)\b.*$", "", prompt, flags=re.IGNORECASE).strip()
    # Strip conversational tails — "but we should...", "and we need...", "we should...", etc.
    prompt = re.sub(r",?\s*\b(but|however)\s+(we|you|i|it)\b.*$", "", prompt, flags=re.IGNORECASE).strip()
    prompt = re.sub(r",?\s*\b(and|,)\s+(we|you|i)\s+(should|need|want|can|could|have)\b.*$", "", prompt, flags=re.IGNORECASE).strip()
    return prompt.strip().rstrip("?.!")


def _build_search_query(text: str) -> str:
    """Build a search query by stripping command words and email addresses."""
    # Remove common command prefixes
    q = re.sub(
        r"^(can you|could you|please|hey vox|vox)\s+",
        "", text.strip(), flags=re.IGNORECASE,
    )
    # Remove search/research command words
    q = re.sub(
        r"^(search\s+for|search|look\s*up|find|google|research|look\s+into|find\s+out|tell\s+me\s+about|explain\s+what)\s+",
        "", q, flags=re.IGNORECASE,
    )
    # Remove email-related tail ("...email me at foo@bar.com...")
    q = re.sub(r"\b(and\s+)?(can you\s+)?email\b.*$", "", q, flags=re.IGNORECASE).strip()
    # Remove trailing punctuation
    q = q.rstrip("?.!")
    return q


# Register intent patterns
_add_pattern(
    r"weather|forecast|temperature|rain|sunny|snow",
    "get_weather",
    lambda m, t: {},
    "Let me check the forecast for you...",
)
_add_pattern(
    r"what time|current time|what.s the time|what is the time|what is the date|\bthe date\b.*\btoday\b|\btoday.s date\b",
    "get_current_time",
    lambda m, t: {},
    "The time right now is",
)
_add_pattern(
    r"system info|gpu|vram|memory|cpu info",
    "get_system_info",
    lambda m, t: {},
    "Let me check the system...",
)
_add_pattern(
    r"\b(screenshot|screen\s*shot|screen\s*cap|capture\s+(my\s+)?screen|print\s*screen)\b"
    r"|\b(what.s on my screen|read my screen)\b",
    "take_screenshot",
    lambda m, t: {},
    "Let me grab that screenshot...",
)
# System commands
_add_pattern(
    r"\b(disk\s*space|storage|how\s+much\s+(space|storage)|drive\s+space)\b",
    "run_command",
    lambda m, t: {"command": "disk_space"},
    "Let me check the drives...",
)
_add_pattern(
    r"\b(what.s using\s+(my\s+)?gpu|gpu\s+process|nvidia.smi)\b",
    "run_command",
    lambda m, t: {"command": "gpu_processes"},
    "Let me check GPU usage...",
)
_add_pattern(
    r"\b(restart\s+ollama|reload\s+ollama)\b",
    "run_command",
    lambda m, t: {"command": "restart_ollama"},
    "Restarting Ollama...",
)
_add_pattern(
    r"\b(network|ip\s*config|ip\s+address|my\s+ip)\b",
    "run_command",
    lambda m, t: {"command": "network_status"},
    "Let me check the network...",
)
# Reminders — must be before Notes to avoid "remind me" matching add_note
_add_pattern(
    r"\b(remind\s+me|set\s+(a|an)\s*(reminder|timer|alarm))\b",
    "set_reminder",
    lambda m, t: {"_raw": t},
    "I'll remind you!",
)
_add_pattern(
    r"\b(show|list|what\s+are)\s+(my\s+)?(reminders|timers|alarms)\b",
    "list_reminders",
    lambda m, t: {},
    "Let me check your reminders...",
)
# Notes / To-Do
_add_pattern(
    r"\b(take\s+a\s+note|add\s+a?\s*note|note\s*:\s*|add\s+to\s+(my\s+)?(to.?do|list))\b",
    "add_note",
    lambda m, t: {"text": re.sub(
        r"^.*?(note\s*:?\s*|to.?do\s*:?\s*|remind\s+me\s*(to\s+)?|add\s+to\s+(my\s+)?(to.?do|list)\s*:?\s*)",
        "", t, flags=re.IGNORECASE,
    ).strip()},
    "Got it, noting that down...",
)
_add_pattern(
    r"\b(what\s+are\s+my\s+notes|show\s+(me\s+)?(my\s+)?notes|my\s+to.?do|list\s+(my\s+)?notes)\b",
    "list_notes",
    lambda m, t: {},
    "Let me check your notes...",
)
_add_pattern(
    r"\b(mark|complete|done|finish|check\s+off)\b.*\b(note|task|to.?do|#?\d+)\b",
    "complete_note",
    lambda m, t: {"note_id": int(re.search(r"#?(\d+)", t).group(1)) if re.search(r"#?(\d+)", t) else 0},
    "Marking that as done...",
)
# Contacts / Address Book
_add_pattern(
    r"\b(add\s+contact|new\s+contact|save\s+contact|contact\s*:\s*)\b",
    "add_contact",
    lambda m, t: {"_raw": t},
    "Adding that contact...",
)
_add_pattern(
    r"\b(what('?s| is)\s+\w+('?s)?\s+(phone|number|email|address))\b",
    "lookup_contact",
    lambda m, t: {"query": re.sub(
        r"\b(what('?s| is)|phone|number|email|address|do you (have|know))\b",
        "", t, flags=re.IGNORECASE,
    ).strip().strip("'s?")},
    "Let me look that up...",
)
_add_pattern(
    r"\b(show|list|who'?s?\s+in)\s+(my\s+)?(contacts|address\s*book)\b",
    "list_contacts",
    lambda m, t: {},
    "Here are your contacts...",
)
_add_pattern(
    r"\b(remove|delete)\s+(contact|#?\d+\s+from\s+contacts)\b",
    "remove_contact",
    lambda m, t: {"contact_id": int(re.search(r"#?(\d+)", t).group(1)) if re.search(r"#?(\d+)", t) else 0},
    "Removing that contact...",
)

# Macros
_add_pattern(
    r"\b(run|execute|do)\s+(my\s+)?(macro|routine)\s+['\"]?(\w[\w\s]+)",
    "run_macro",
    lambda m, t: {"name": re.sub(
        r"^(run|execute|do)\s+(my\s+)?(macro|routine)\s+['\"]?",
        "", t, flags=re.IGNORECASE,
    ).strip().strip("'\"")},
    "Running your macro...",
)
_add_pattern(
    r"\b(create|add|save|define|set\s*up)\s+(a\s+)?(macro|routine)\b",
    "add_macro",
    lambda m, t: {"_raw": t},
    "Setting up your macro...",
)
_add_pattern(
    r"\b(show|list|what\s+are)\s+(my\s+)?(macros|routines)\b",
    "list_macros",
    lambda m, t: {},
    "Here are your macros...",
)
_add_pattern(
    r"\b(delete|remove)\s+(my\s+)?(macro|routine)\b",
    "remove_macro",
    lambda m, t: {"name": re.sub(
        r"^.*?(macro|routine)\s+['\"]?",
        "", t, flags=re.IGNORECASE,
    ).strip().strip("'\"")},
    "Removing that macro...",
)


# File navigator
_add_pattern(
    r"\b(what('?s| is)\s+in\s+(my\s+)?(downloads|documents|desktop))\b",
    "list_files",
    lambda m, t: {"directory": m.group(4).lower()},
    "Let me check that folder...",
)
_add_pattern(
    r"\b(find|search\s+for|look\s+for|where('?s| is))\s+(that\s+|the\s+|my\s+)?\S*\.(pdf|docx?|xlsx?|txt|csv|png|jpg|zip)\b",
    "find_file",
    lambda m, t: {"pattern": re.search(r"\S+\.\S+", t).group(0)
                  if re.search(r"\S+\.\S+", t) else ""},
    "Searching for that file...",
)
_add_pattern(
    r"\b(latest|newest|recent|last)\s+(download|file|document)\b",
    "list_files",
    lambda m, t: {"directory": "downloads", "sort": "newest"},
    "Checking your latest files...",
)

# Daily briefing
_add_pattern(
    r"\b(good\s+morning|morning\s+briefing|daily\s+briefing|briefing|brief\s+me)\b",
    "daily_briefing",
    lambda m, t: {},
    "Good morning! Let me get your briefing ready...",
)

# News / RSS feeds
_add_pattern(
    r"\b(news|headlines|rss|feed)\b",
    "get_news",
    lambda m, t: {"topic": re.sub(
        r"\b(what.s|what is|show me|give me|get|the|latest|today.s|in the|news|headlines)\b",
        "", t, flags=re.IGNORECASE,
    ).strip() or "general"},
    "Checking the news...",
)

# Document search (RAG)
_add_pattern(
    r"\b(what\s+does|find|search)\b.*\b(document|file|pdf|lease|manual|contract|report)\b",
    "search_documents",
    lambda m, t: {"query": t},
    "Searching your documents...",
)
_add_pattern(
    r"\b(index|scan|reindex)\s+(my\s+)?(documents|files|folder)\b",
    "index_documents",
    lambda m, t: {},
    "Indexing your documents...",
)

# Home Assistant / smart home
_add_pattern(
    r"\b(turn|switch)\s+(on|off)\s+(the\s+)?(.+?)(\s+light|\s+lamp|\s+fan|\s+switch)?\s*$",
    "smart_home",
    lambda m, t: {"action": "toggle", "entity_hint": m.group(4).strip(), "state": m.group(2)},
    "On it...",
)
_add_pattern(
    r"\b(set|change)\s+(the\s+)?thermostat\s+to\s+(\d+)",
    "smart_home",
    lambda m, t: {"action": "climate", "temperature": int(m.group(3))},
    "Adjusting the thermostat...",
)
_add_pattern(
    r"\b(lock|unlock)\s+(the\s+)?(.+?)(\s+door|\s+lock)?\s*$",
    "smart_home",
    lambda m, t: {"action": m.group(1).lower(), "entity_hint": m.group(3).strip()},
    "Done.",
)
_add_pattern(
    r"\bis\s+the\s+(.+?)\s+(open|closed|on|off|locked|unlocked)\s*\??",
    "smart_home",
    lambda m, t: {"action": "status", "entity_hint": m.group(1).strip()},
    "Let me check...",
)
_add_pattern(
    r"\b(activate|set)\s+(.+?)\s+(scene|mode)\b|\b(.+?)\s+mode\b",
    "smart_home",
    lambda m, t: {"action": "scene", "scene_name": (m.group(2) or m.group(4) or "").strip()},
    "Activating scene...",
)


def _parse_volume_action(text: str) -> str:
    t = text.lower()
    if "mute" in t and "unmute" not in t:
        return "mute"
    if "unmute" in t:
        return "unmute"
    if any(w in t for w in ("up", "louder")):
        return "volume_up"
    if any(w in t for w in ("down", "quieter", "softer")):
        return "volume_down"
    return "volume_up"


# Media control
_add_pattern(
    r"\b(play|resume)\s+(some\s+)?music\b|\bplay\s+(some\s+)?(?:rock|jazz|classical|chill|lo-?fi)\b",
    "media_control",
    lambda m, t: {"action": "play"},
    "Playing music...",
)
_add_pattern(
    r"\b(pause|stop)\s+(the\s+)?music\b|\bpause\b",
    "media_control",
    lambda m, t: {"action": "pause"},
    "Pausing...",
)
_add_pattern(
    r"\b(skip|next)\s+(this\s+)?(song|track)\b|\bnext\s+song\b|\bskip\b",
    "media_control",
    lambda m, t: {"action": "next"},
    "Skipping to the next track...",
)
_add_pattern(
    r"\bprevious\s+(song|track)\b|\bgo\s+back\s+(a\s+)?(song|track)\b",
    "media_control",
    lambda m, t: {"action": "previous"},
    "Going back...",
)
_add_pattern(
    r"\b(what.?s|what\s+is)\s+(playing|this\s+song)\b|\bcurrent\s+(song|track)\b",
    "media_control",
    lambda m, t: {"action": "now_playing"},
    "Let me check...",
)
_add_pattern(
    r"\b(volume)\s+(up|down|louder|quieter|softer|mute|unmute)\b"
    r"|\b(louder|quieter|softer|mute|unmute)\b",
    "media_control",
    lambda m, t: {"action": _parse_volume_action(t)},
    "Adjusting volume...",
)

# Image upscaling
_add_pattern(
    r"\b(upscale|enhance|enlarge|upres|super\s*res|higher\s*res)\b.*\b(image|photo|picture|pic)\b",
    "upscale_image",
    lambda m, t: {"path": re.search(r"\S+\.(?:png|jpg|jpeg|webp)", t).group(0)
                  if re.search(r"\S+\.(?:png|jpg|jpeg|webp)", t) else ""},
    "Upscaling that image...",
)
_add_pattern(
    r"\bmake\s+(?:this|that|the)\s+(?:image|photo|picture|pic)\s+(?:bigger|larger|higher\s+res)",
    "upscale_image",
    lambda m, t: {"path": re.search(r"\S+\.(?:png|jpg|jpeg|webp)", t).group(0)
                  if re.search(r"\S+\.(?:png|jpg|jpeg|webp)", t) else ""},
    "Upscaling that image...",
)

# Code / calculations
_add_pattern(
    r"\b(calculate|compute|what\s+is\s+\d|convert|how\s+much\s+is)\b",
    "run_code",
    lambda m, t: {"expression": re.sub(
        r"^(calculate|compute|what\s+is|convert|how\s+much\s+is)\s*",
        "", t, flags=re.IGNORECASE,
    ).strip()},
    "Let me calculate that...",
)
_add_pattern(
    r"\b(run|execute)\s+(this\s+)?(python|code|script)\b",
    "run_code",
    lambda m, t: {"code": re.sub(
        r"^(run|execute)\s+(this\s+)?(python|code|script)\s*:?\s*",
        "", t, flags=re.IGNORECASE,
    ).strip()},
    "Running that code...",
)

# Clipboard
_add_pattern(
    r"\b(what('?s| is)\s+on\s+(my\s+)?clipboard|read\s+(my\s+)?clipboard|paste\s+from\s+clipboard)\b",
    "read_clipboard",
    lambda m, t: {},
    "Let me check your clipboard...",
)
_add_pattern(
    r"\b(copy|put)\s+(that|this|it)\s+(to|on|in)\s+(my\s+)?clipboard\b",
    "write_clipboard",
    lambda m, t: {},
    "Copied to clipboard!",
)

def _build_search_args(m, t):
    """Build search args — detect deep search intent from keywords."""
    deep = bool(re.search(
        r"\b(deep\s*search|research|in\s*depth|thorough|comprehensive|detailed)\b",
        t, re.IGNORECASE,
    ))
    return {"query": _build_search_query(t), "deep": deep}

_add_pattern(
    r"\b(search|look\s*up|find|google|search\s+for|research)\b",
    "web_search",
    _build_search_args,
    "Let me search for that...",
)
_add_pattern(
    r"\b(download|fetch|open|get|grab)\b.*\b(pdf|page|url|link|site|website)\b",
    "web_fetch",
    lambda m, t: {"url": _extract_url(t)},
    "Let me fetch that for you...",
)
_add_pattern(
    r"https?://\S+",
    "web_fetch",
    lambda m, t: {"url": _extract_url(t)},
    "Let me fetch that for you...",
)
# Map / satellite / location requests — BEFORE image gen so real addresses
# don't get routed to AI image generation.
_add_pattern(
    r"\b(satellite|aerial|map|street\s*view|bird.?s?\s*eye|top\s*down)\s*(view|image|photo|picture|map)?\s*(of|for)?\b"
    r"|\b(directions|navigate|route)\s*(to|from)\b"
    r"|\b(show|get|pull\s*up|find)\s+(me\s+)?(a\s+)?(map|satellite|aerial)\b",
    "get_map",
    lambda m, t: {"location": _extract_location(t)},
    "Let me pull up that location...",
)
# Selfie / persona-aware image triggers — HIGHEST PRIORITY for image generation.
# Must come BEFORE email patterns so "send me a selfie" routes to generate_image first,
# with send_email chained as secondary.
_add_pattern(
    r"\b(selfie|selfy)\b"
    r"|\b(picture|pic|photo|image)\s+(of\s+)?(you|yourself)"
    r"|\bwhat\s+do\s+you\s+look\s+like\b"
    r"|\bshow\s+(me\s+)?(yourself|what\s+you\s+look\s+like)"
    r"|\btake\s+a\s+(pic|picture|photo|selfie|snap|shot)\b"
    r"|\b(send|email|mail|give)\s+me\s+a\s+(selfie|pic|picture|photo)"
    r"|\blet\s+me\s+see\s+you\b",
    "generate_image",
    lambda m, t: {
        "prompt": _build_persona_prompt(t),
        "_selfie": True,
        "_nsfw_unlocked": bool(re.search(r"\bohwx\s+\w+\b", t, re.IGNORECASE)),
    },
    "Let me take a pic for you...",
)
# LoRA trigger word "ohwx <name>" — ANY message containing this is a selfie request.
# The trigger word IS the intent — user expects persona image generation.
_add_pattern(
    r"\bohwx\s+\w+\b",
    "generate_image",
    lambda m, t: {
        "prompt": _build_persona_prompt(t),
        "_selfie": True,
        "_nsfw_unlocked": True,
    },
    "Let me take a pic for you...",
)
# Email with explicit address (highest priority for email)
_add_pattern(
    r"\b(email|mail)\b.*\b\S+@\S+\.\S+",
    "send_email",
    lambda m, t: {"to": _extract_email(t)},
    "I'll send that over...",
)
# Image generation patterns — BEFORE generic "email/mail me" so
# "email me a picture" routes to generate_image (not send_email)
_add_pattern(
    r"\b(generate|create|draw|make|paint|imagine)\b.*\b(image|picture|photo|artwork|illustration)\b",
    "generate_image",
    lambda m, t: {"prompt": _extract_image_prompt(t), "count": _extract_image_count(t)},
    "Let me generate that image for you...",
)
_add_pattern(
    r"\b(draw|paint)\s+me\b",
    "generate_image",
    lambda m, t: {"prompt": _extract_image_prompt(t)},
    "Let me generate that image for you...",
)
_add_pattern(
    r"\bimagine\b",
    "generate_image",
    lambda m, t: {"prompt": _extract_image_prompt(t)},
    "Let me generate that image for you...",
)
# "email/mail/send/give me a picture of X" — implies generation
_add_pattern(
    r"\b(email|mail|send|give|show)\s+me\b.*\b(image|picture|photo|pic|pics)\b",
    "generate_image",
    lambda m, t: {"prompt": _extract_image_prompt(t)},
    "Let me generate that image for you...",
)
# "give me N pictures of X" / "show me X" — broader image triggers
_add_pattern(
    r"\b(give|show|get)\s+me\b.*\b(picture|image|photo|pic|pics)\b",
    "generate_image",
    lambda m, t: {"prompt": _extract_image_prompt(t)},
    "Let me generate that image for you...",
)
# "\d+ pictures/images of" — count-based trigger
_add_pattern(
    r"\b\d+\s+(picture|image|photo|pic|pics)\w*\s+(of|with)\b",
    "generate_image",
    lambda m, t: {"prompt": _extract_image_prompt(t), "count": _extract_image_count(t)},
    "Let me generate those images for you...",
)
# Broad "show me a [description]" — lowest-priority image trigger.
# Catches requests like "show me a cat on a beach" without requiring
# an explicit image noun (picture/photo/etc).
_add_pattern(
    r"\bshow\s+me\s+(a|an|some|the)\b",
    "generate_image",
    lambda m, t: {"prompt": _extract_image_prompt(t)},
    "Let me generate that image for you...",
)
# Mature content + action verb — implicit image request routed to unrestricted model.
# Uses same keyword list as _NSFW_KEYWORDS (loaded from config file).
_add_pattern(
    r"\b(show|give|make|get|draw|paint|create|generate)\b.*" + _NSFW_KEYWORDS.pattern,
    "generate_image",
    lambda m, t: {
        "prompt": _extract_image_prompt(t),
        "_nsfw_unlocked": bool(re.search(r"\bohwx\s+\w+\b", t, re.IGNORECASE)),
    },
    "Let me generate that image for you...",
)
# Generic "email/mail me" / "send me" (no address) — AFTER image patterns
_add_pattern(
    r"\b(email|mail|send)\s+(me|it|this|that|the)\b",
    "send_email",
    lambda m, t: {"to": _extract_email(t)},
    "I'll send that over...",
)


_NEGATION_PATTERN = re.compile(
    r"\b(don'?t|do\s+not|don't|no|never|stop|skip|without|instead\s+of|rather\s+than)\b"
    r".*\b(image|picture|photo|pic|selfie|generate|draw|create|show)\b",
    re.IGNORECASE,
)

_DESCRIBE_PATTERN = re.compile(
    r"\b(describe|tell\s+me|explain|say|words|text)\b"
    r".*\b(you|yourself|look\s+like|appearance)\b",
    re.IGNORECASE,
)


def _should_suppress_image(text: str) -> bool:
    """Check if the user is explicitly asking NOT to generate an image."""
    return bool(_NEGATION_PATTERN.search(text) or _DESCRIBE_PATTERN.search(text))


def _is_negated(text: str, tool_name: str) -> bool:
    """Check if the user is explicitly asking NOT to use this tool.

    Catches patterns like "don't search for that", "I don't need the weather",
    "no email", "skip the image", "stop generating pictures".
    """
    # Tool-specific keywords that follow the negation
    _TOOL_KEYWORDS: dict[str, list[str]] = {
        "get_weather": ["weather", "forecast", "temperature"],
        "get_current_time": ["time", "date", "clock"],
        "get_system_info": ["system", "gpu", "vram", "cpu"],
        "web_search": ["search", "look up", "google", "find"],
        "web_fetch": ["fetch", "download", "grab"],
        "send_email": ["email", "mail", "send"],
        "generate_image": ["image", "picture", "photo", "pic", "selfie",
                          "generate", "draw", "create", "show"],
        "take_screenshot": ["screenshot", "screen", "capture"],
    }
    keywords = _TOOL_KEYWORDS.get(tool_name, [])
    if not keywords:
        return False

    keyword_pattern = "|".join(re.escape(k) for k in keywords)
    negation = re.compile(
        rf"\b(don'?t|do\s+not|don't|no|never|stop|skip|don't\s+need|no\s+need)\b"
        rf".*\b({keyword_pattern})\b",
        re.IGNORECASE,
    )
    return bool(negation.search(text))


_REFRESH_PATTERN = re.compile(
    r"\b(refresh|force\s*refresh|update|re-?check|fresh)\b",
    re.IGNORECASE,
)


def detect_intent(text: str) -> DetectedIntent | None:
    """Fast intent detection via regex. Returns first match or None."""
    global _last_tool, _last_tool_args

    # "refresh" busts cache for the next tool call
    if _REFRESH_PATTERN.search(text):
        cache_bust()

    # Check for follow-up requests ("another one", "again", "one more")
    if _last_tool and _FOLLOWUP_PATTERN.search(text.strip()):
        log.info("Follow-up detected — repeating %s", _last_tool)
        return DetectedIntent(
            tool_name=_last_tool,
            args=dict(_last_tool_args),
            bridge_phrase="Coming right up...",
        )

    suppress_image = _should_suppress_image(text)
    for pattern, tool_name, arg_builder, bridge in _INTENT_PATTERNS:
        if suppress_image and tool_name == "generate_image":
            continue
        if _is_negated(text, tool_name):
            continue
        match = pattern.search(text)
        if match:
            args = arg_builder(match, text)
            intent = DetectedIntent(
                tool_name=tool_name,
                args=args,
                bridge_phrase=bridge,
            )
            # Track for follow-up
            _last_tool = tool_name
            _last_tool_args = dict(args)
            log.info("Intent detected: %s args=%s", tool_name, intent.args)
            return intent
    # Check if text matches a saved macro trigger phrase
    try:
        from vox.macros import find_macro
        macro = find_macro(text)
        if macro:
            log.info("Macro match: '%s'", macro["name"])
            return DetectedIntent(
                tool_name="run_macro",
                args={"name": macro["name"]},
                bridge_phrase=f"Running {macro['name']}...",
            )
    except Exception:
        log.debug("Macro lookup skipped")

    log.debug("No intent detected for: %s", text[:80])
    return None


def detect_all_intents(text: str) -> list[DetectedIntent]:
    """Detect ALL matching intents in the text (for tool chaining)."""
    intents = []
    seen = set()
    suppress_image = _should_suppress_image(text)
    for pattern, tool_name, arg_builder, bridge in _INTENT_PATTERNS:
        if tool_name in seen:
            continue
        if suppress_image and tool_name == "generate_image":
            continue
        if _is_negated(text, tool_name):
            continue
        match = pattern.search(text)
        if match:
            intents.append(DetectedIntent(
                tool_name=tool_name,
                args=arg_builder(match, text),
                bridge_phrase=bridge,
            ))
            seen.add(tool_name)

    # Suppress send_email when generate_image is primary and user didn't
    # explicitly request email delivery — "send me a selfie" means show it,
    # not email it. But "email me a selfie" or "send in email" keeps it.
    _explicit_email_request = re.compile(
        r"\S+@\S+\.\S+"  # explicit address
        r"|\b(in|via|by|through)\s+e?-?mail\b"  # "in email", "via email"
        r"|^(email|mail)\s+me\b"  # starts with "email me"
        r"|\bemail\s+(me|it|this|that)\b",  # "email me", "email it"
        re.IGNORECASE,
    )
    if (
        intents
        and intents[0].tool_name == "generate_image"
        and any(i.tool_name == "send_email" for i in intents)
        and not _explicit_email_request.search(text)
    ):
        intents = [i for i in intents if i.tool_name != "send_email"]

    if intents:
        log.info("All intents: %s", [i.tool_name for i in intents])
    return intents


# Validation patterns — stricter than intent detection.
# Used to block spurious LLM-initiated tool calls that don't match the user's actual request.
_TOOL_VALIDATORS: dict[str, re.Pattern] = {
    "get_weather": re.compile(
        r"\b(weather|forecast|temperature|rain(?:ing)?|sunny|snow(?:ing)?|storm|humid|wind|cold|hot|warm|cool)\b",
        re.IGNORECASE,
    ),
    "get_current_time": re.compile(
        r"\b(what time|current time|the time|the date|what day|today.s date|what is the time|what is the date)\b",
        re.IGNORECASE,
    ),
    "get_system_info": re.compile(
        r"\b(system info|gpu|vram|memory usage|cpu info|system stats|hardware)\b",
        re.IGNORECASE,
    ),
    "web_search": re.compile(
        r"\b(search|look\s*up|find|google|lookup|research)\b",
        re.IGNORECASE,
    ),
    "web_fetch": re.compile(
        r"\b(download|fetch|open|get|grab|pdf|page|url|link|site|website)\b|https?://\S+",
        re.IGNORECASE,
    ),
    "get_map": re.compile(
        r"\b(satellite|aerial|map|street\s*view|bird.?s?\s*eye|top\s*down|directions|navigate|route|address)\b",
        re.IGNORECASE,
    ),
    "send_email": re.compile(
        r"\b(email|send|mail)\b.*(\S+@\S+\.\S+|\b(me|it|this|that|the|results|report)\b)",
        re.IGNORECASE,
    ),
    "generate_image": re.compile(
        r"\b(generate|create|draw|make|paint|imagine)\b.*\b(image|picture|photo|artwork|illustration)\b"
        r"|\b(draw|paint)\s+me\b"
        r"|\bimagine\b"
        r"|\b(email|mail|send|give|show)\s+me\b.*\b(image|picture|photo|pic|pics|selfie)\b"
        r"|\b\d+\s+(picture|image|photo|pic|pics)\w*\s+(of|with)\b"
        r"|\b(selfie|selfy)\b"
        r"|\b(picture|pic|photo|image)\s+(of\s+)?(you|yourself)"
        r"|\bwhat\s+do\s+you\s+look\s+like\b"
        r"|\btake\s+a\s+(pic|picture|photo|selfie|snap|shot)\b"
        r"|\blet\s+me\s+see\s+you\b"
        r"|\bshow\s+me\s+(a|an|some|the)\b",
        re.IGNORECASE,
    ),
    "take_screenshot": re.compile(
        r"\b(screenshot|screen\s*shot|screen\s*cap|capture\s+(my\s+)?screen|print\s*screen"
        r"|what.s on my screen|read my screen)\b",
        re.IGNORECASE,
    ),
    "list_files": re.compile(
        r"\b(downloads|documents|desktop|files|folder|latest|newest|recent)\b",
        re.IGNORECASE,
    ),
    "find_file": re.compile(
        r"\b(find|search|where|look)\b.*\.\w{2,4}\b",
        re.IGNORECASE,
    ),
    "daily_briefing": re.compile(
        r"\b(morning|briefing|brief\s+me|good\s+morning)\b",
        re.IGNORECASE,
    ),
    "upscale_image": re.compile(
        r"\b(upscale|enhance|enlarge|upres|super\s*res|bigger|larger|higher\s*res)\b",
        re.IGNORECASE,
    ),
    "run_code": re.compile(
        r"\b(calculate|compute|convert|run\s+python|execute\s+code|what\s+is\s+\d|how\s+much)\b",
        re.IGNORECASE,
    ),
    "read_clipboard": re.compile(
        r"\b(clipboard|paste|pasted)\b",
        re.IGNORECASE,
    ),
    "write_clipboard": re.compile(
        r"\b(copy|clipboard)\b",
        re.IGNORECASE,
    ),
    "add_contact": re.compile(
        r"\b(add|new|save|create)\s+(a\s+)?contact\b|contact\s*:",
        re.IGNORECASE,
    ),
    "lookup_contact": re.compile(
        r"\b(phone|number|email|address)\b.*\b\w+\b|\bcontact\s+info\b|\blook\s*up\b",
        re.IGNORECASE,
    ),
    "list_contacts": re.compile(
        r"\b(contacts|address\s*book|show\s+(my\s+)?contacts)\b",
        re.IGNORECASE,
    ),
    "remove_contact": re.compile(
        r"\b(remove|delete)\s+(a\s+)?contact\b",
        re.IGNORECASE,
    ),
    "get_news": re.compile(
        r"\b(news|headlines|rss|feed)\b",
        re.IGNORECASE,
    ),
    "search_documents": re.compile(
        r"\b(document|file|pdf|lease|manual|contract|report|paper|invoice)\b",
        re.IGNORECASE,
    ),
    "index_documents": re.compile(
        r"\b(index|scan|reindex|catalog)\b.*\b(document|file|folder)\b",
        re.IGNORECASE,
    ),
    "smart_home": re.compile(
        r"\b(light|lamp|thermostat|lock|door|garage|fan|switch|scene|mode|turn\s+(on|off)"
        r"|dim|bright|temperature|heat|cool|blind|curtain)\b",
        re.IGNORECASE,
    ),
    "media_control": re.compile(
        r"\b(play|pause|stop|skip|next|previous|volume|louder|quieter|softer|mute|unmute"
        r"|music|song|track|what.?s\s+playing)\b",
        re.IGNORECASE,
    ),
    "set_reminder": re.compile(
        r"\b(remind|reminder|timer|alarm)\b",
        re.IGNORECASE,
    ),
    "list_reminders": re.compile(
        r"\b(reminders?|timers?|alarms?)\b",
        re.IGNORECASE,
    ),
}


def validate_tool_call(tool_name: str, user_message: str) -> bool:
    """Check if a tool call is actually relevant to the user's current message.

    This catches cases where the LLM hallucinates tool calls based on
    conversation history rather than the current request.
    """
    validator = _TOOL_VALIDATORS.get(tool_name)
    if validator is None:
        # Unknown tool — let it through (execute_tool will handle the error)
        return True
    return bool(validator.search(user_message))


# ---------------------------------------------------------------------------
# Tool definitions (Ollama format) — used as fallback when intent not detected
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_info",
            "description": "Get system information (GPU, memory, CPU)",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name (defaults to auto-detect)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information. Use when the user asks to find something.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch a URL and return its content. For HTML pages, returns extracted text. For PDFs, downloads and saves the file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_map",
            "description": "Get a real satellite or map image of a location/address. Use for addresses, places, directions, 'satellite view of', 'map of'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Address or place name to look up",
                    },
                    "map_type": {
                        "type": "string",
                        "description": "Map type: satellite, roadmap, terrain, hybrid",
                        "enum": ["satellite", "roadmap", "terrain", "hybrid"],
                    },
                    "zoom": {
                        "type": "integer",
                        "description": "Zoom level (1=world, 18=street, 20=building)",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate an AI image from a text prompt using Stable Diffusion. Do NOT use for real locations or addresses — use get_map instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the image to generate",
                    },
                    "style": {
                        "type": "string",
                        "description": "Optional style modifier (e.g. 'photorealistic', 'watercolor', 'oil painting')",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to a recipient. Use when the user asks to email something to an address.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line",
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body content",
                    },
                    "attachments": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of file paths to attach",
                    },
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a safe whitelisted system command (disk space, GPU usage, network status, etc.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to run: disk_space, gpu_processes, network_status, restart_ollama",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "take_screenshot",
            "description": "Take a screenshot of the user's screen",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_note",
            "description": "Save a note or to-do item for the user",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The note text",
                    },
                    "category": {
                        "type": "string",
                        "description": "Category (general, todo, shopping, etc.)",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_notes",
            "description": "Show the user's notes and to-do items",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_note",
            "description": "Mark a note or task as done",
            "parameters": {
                "type": "object",
                "properties": {
                    "note_id": {
                        "type": "integer",
                        "description": "The note number to mark as done",
                    },
                },
                "required": ["note_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Set a reminder for a future time",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "What to remind about"},
                    "minutes": {
                        "type": "integer",
                        "description": "Minutes from now",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_reminders",
            "description": "Show all pending reminders",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a safe directory (downloads, documents, desktop)",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Which folder: downloads, documents, or desktop",
                    },
                    "sort": {
                        "type": "string",
                        "description": "Sort order: newest, oldest, largest, name",
                    },
                },
                "required": ["directory"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_file",
            "description": "Search for a file by name pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Filename or glob pattern to search for",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "daily_briefing",
            "description": "Get a morning briefing with weather, notes, and time",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Get news headlines and summaries from RSS feeds",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "News topic: general, tech, science, world, business",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "media_control",
            "description": "Control media playback — play, pause, skip, volume",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action: play, pause, next, previous, now_playing, volume_up, volume_down, mute, unmute",
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search indexed documents (PDFs, text files) for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for in your documents"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "index_documents",
            "description": "Index local documents for search (PDFs, text files, code)",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Directory to index (uses RAG_DOCS_DIR if not specified)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "smart_home",
            "description": "Control smart home devices via Home Assistant — lights, thermostat, locks, scenes",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action: toggle, climate, lock, unlock, status, scene",
                    },
                    "entity_hint": {
                        "type": "string",
                        "description": "Natural language hint for the entity (e.g., 'living room lights')",
                    },
                    "state": {"type": "string", "description": "on/off for toggle actions"},
                    "temperature": {"type": "integer", "description": "Target temperature for climate"},
                    "scene_name": {"type": "string", "description": "Scene name to activate"},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "upscale_image",
            "description": "Upscale an image to higher resolution",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to image file to upscale",
                    },
                    "scale": {
                        "type": "integer",
                        "description": "Scale factor (2 or 4, default 2)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": "Run a Python expression or code snippet safely",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate",
                    },
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_macro",
            "description": "Run a saved macro/routine by name",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Macro name"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_macros",
            "description": "Show all saved macros/routines",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_clipboard",
            "description": "Read the contents of the system clipboard",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_clipboard",
            "description": "Write text to the system clipboard",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to copy"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_contact",
            "description": "Add a new contact to the address book",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Contact name"},
                    "email": {"type": "string", "description": "Email address"},
                    "phone": {"type": "string", "description": "Phone number"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags/groups (e.g. family, work, team)",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_contact",
            "description": "Look up a contact by name to get their info",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Name to search for"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_contacts",
            "description": "Show all contacts in the address book",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_contact",
            "description": "Remove a contact from the address book",
            "parameters": {
                "type": "object",
                "properties": {
                    "contact_id": {
                        "type": "integer",
                        "description": "The contact ID to remove",
                    },
                },
                "required": ["contact_id"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

_TOOL_REGISTRY: dict[str, callable] = {}


def _register(name: str):
    def decorator(fn):
        _TOOL_REGISTRY[name] = fn
        return fn

    return decorator


@_register("get_current_time")
def _get_current_time(**kwargs) -> str:
    now = datetime.datetime.now()
    return now.strftime("%A, %B %d, %Y at %I:%M %p")


@_register("get_system_info")
def _get_system_info(**kwargs) -> str:
    import platform

    lines = [
        f"OS: {platform.system()} {platform.release()}",
        f"Python: {platform.python_version()}",
    ]
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            lines.append(f"GPU: {gpu} ({vram:.1f} GB VRAM)")
    except ImportError:
        lines.append("GPU: torch not available")
    return "\n".join(lines)


@_register("get_weather")
def _get_weather(**kwargs) -> str:
    """Fetch weather from Open-Meteo API (free, no key needed)."""
    import json
    import urllib.request

    try:
        # Step 1: Get location from IP (free, no key)
        geo_url = "https://ipapi.co/json/"
        with urllib.request.urlopen(geo_url, timeout=3) as resp:
            geo = json.loads(resp.read())
        lat, lon = geo.get("latitude", 40.71), geo.get("longitude", -74.01)
        city = geo.get("city", "Unknown")

        # Step 2: Get forecast from Open-Meteo (free, no key)
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max,weathercode"
            f"&temperature_unit=fahrenheit"
            f"&timezone=auto"
            f"&forecast_days=7"
        )
        with urllib.request.urlopen(weather_url, timeout=5) as resp:
            data = json.loads(resp.read())

        daily = data.get("daily", {})
        dates = daily.get("time", [])
        highs = daily.get("temperature_2m_max", [])
        lows = daily.get("temperature_2m_min", [])
        rain = daily.get("precipitation_probability_max", [])
        codes = daily.get("weathercode", [])

        code_map = {
            0: "Clear", 1: "Mostly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
            61: "Light rain", 63: "Rain", 65: "Heavy rain",
            71: "Light snow", 73: "Snow", 75: "Heavy snow",
            80: "Rain showers", 81: "Rain showers", 82: "Heavy rain showers",
            95: "Thunderstorm",
        }

        lines = [f"7-day forecast for {city}:"]
        for i in range(min(7, len(dates))):
            condition = code_map.get(codes[i], f"Code {codes[i]}")
            lines.append(
                f"  {dates[i]}: {condition}, High {highs[i]:.0f}F / Low {lows[i]:.0f}F, "
                f"{rain[i]}% chance of rain"
            )
        return "\n".join(lines)

    except Exception as e:
        return f"Weather lookup failed: {e}"


# ---------------------------------------------------------------------------
# Notes / To-Do tool — JSON-backed, local-only
# ---------------------------------------------------------------------------

_NOTES_FILE = None

def _get_notes_file():
    global _NOTES_FILE
    if _NOTES_FILE is None:
        from vox.config import PROJECT_ROOT
        data_dir = PROJECT_ROOT / "data"
        data_dir.mkdir(exist_ok=True)
        _NOTES_FILE = data_dir / "notes.json"
    return _NOTES_FILE


def _load_notes() -> list[dict]:
    import json
    path = _get_notes_file()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def _save_notes(notes: list[dict]):
    import json
    path = _get_notes_file()
    path.write_text(json.dumps(notes, indent=2, default=str), encoding="utf-8")


@_register("add_note")
def _add_note(text: str = "", category: str = "general", **kwargs) -> str:
    """Add a note or to-do item."""
    if not text:
        return "No note text provided."
    notes = _load_notes()
    note = {
        "id": len(notes) + 1,
        "text": text,
        "category": category,
        "done": False,
        "created": datetime.datetime.now().isoformat(),
    }
    notes.append(note)
    _save_notes(notes)
    log.info("Note added: #%d %s", note["id"], text[:50])
    return f"Got it! Note #{note['id']} saved: {text}"


@_register("list_notes")
def _list_notes(category: str = "", **kwargs) -> str:
    """List all notes, optionally filtered by category."""
    notes = _load_notes()
    if not notes:
        return "No notes yet."
    if category:
        notes = [n for n in notes if n.get("category", "").lower() == category.lower()]
        if not notes:
            return f"No notes in category '{category}'."
    lines = []
    for n in notes:
        status = "DONE" if n.get("done") else "TODO"
        lines.append(f"#{n['id']} [{status}] {n['text']}")
    return "Your notes:\n" + "\n".join(lines)


@_register("complete_note")
def _complete_note(note_id: int = 0, **kwargs) -> str:
    """Mark a note as done."""
    if not note_id:
        return "Which note? Give me the note number."
    notes = _load_notes()
    for n in notes:
        if n["id"] == note_id:
            n["done"] = True
            _save_notes(notes)
            return f"Done! Marked #{note_id} as complete."
    return f"Note #{note_id} not found."


# ---------------------------------------------------------------------------
# Contacts / Address Book
# ---------------------------------------------------------------------------

def _parse_contact_raw(raw: str) -> dict:
    """Parse free-form contact text like 'Mike, mechanic, 555-1234, mike@email.com'."""
    parts = [p.strip() for p in re.split(r"[,;]", raw) if p.strip()]
    # Strip command prefix
    parts_clean = []
    for p in parts:
        cleaned = re.sub(
            r"^(add|new|save)\s+(a\s+)?contact\s*:?\s*",
            "", p, flags=re.IGNORECASE,
        ).strip()
        if cleaned:
            parts_clean.append(cleaned)
    result = {"name": "", "email": "", "phone": "", "tags": []}
    for p in parts_clean:
        if re.search(r"@\S+\.\S+", p):
            result["email"] = p
        elif re.search(r"\d{3}[\s.-]?\d{3,4}[\s.-]?\d{4}", p):
            result["phone"] = p
        elif not result["name"]:
            result["name"] = p
        else:
            result["tags"].append(p)
    return result


@_register("add_contact")
def _add_contact_tool(name: str = "", email: str = "", phone: str = "",
                      tags: list | None = None, _raw: str = "", **kwargs) -> str:
    """Add a contact to the address book."""
    from vox.contacts import add_contact
    if _raw and not name:
        parsed = _parse_contact_raw(_raw)
        name = parsed["name"]
        email = email or parsed["email"]
        phone = phone or parsed["phone"]
        tags = tags or parsed["tags"] or None
    if not name:
        return "I need at least a name for the contact."
    contact = add_contact(name=name, email=email, phone=phone, tags=tags)
    parts = [f"#{contact['id']} {name}"]
    if email:
        parts.append(f"email: {email}")
    if phone:
        parts.append(f"phone: {phone}")
    return f"Contact saved! {', '.join(parts)}"


@_register("lookup_contact")
def _lookup_contact(query: str = "", **kwargs) -> str:
    """Look up a contact by name."""
    from vox.contacts import lookup
    if not query:
        return "Who are you looking for?"
    matches = lookup(query)
    if not matches:
        return f"No contacts matching '{query}'."
    lines = []
    for c in matches[:5]:
        info = [c["name"]]
        if c.get("email"):
            info.append(f"email: {c['email']}")
        if c.get("phone"):
            info.append(f"phone: {c['phone']}")
        if c.get("tags"):
            info.append(f"tags: {', '.join(c['tags'])}")
        lines.append(" | ".join(info))
    return "\n".join(lines)


@_register("list_contacts")
def _list_contacts(**kwargs) -> str:
    """List all contacts."""
    from vox.contacts import list_all
    contacts = list_all()
    if not contacts:
        return "No contacts yet. Say 'add contact: Name, email, phone' to add one."
    lines = []
    for c in contacts:
        info = [f"#{c['id']} {c['name']}"]
        if c.get("email"):
            info.append(c["email"])
        if c.get("phone"):
            info.append(c["phone"])
        if c.get("tags"):
            info.append(f"[{', '.join(c['tags'])}]")
        lines.append(" | ".join(info))
    return f"{len(contacts)} contacts:\n" + "\n".join(lines)


@_register("remove_contact")
def _remove_contact(contact_id: int = 0, **kwargs) -> str:
    """Remove a contact."""
    from vox.contacts import remove_contact
    if not contact_id:
        return "Which contact? Give me the contact ID number."
    if remove_contact(contact_id):
        return f"Contact #{contact_id} removed."
    return f"Contact #{contact_id} not found."


# ---------------------------------------------------------------------------
# Macros — user-defined tool chains
# ---------------------------------------------------------------------------

@_register("run_macro")
def _run_macro(name: str = "", **kwargs) -> str:
    """Run a saved macro by name."""
    from vox.macros import find_macro, execute_macro
    if not name:
        return "Which macro? Tell me the name."
    macro = find_macro(name)
    if not macro:
        return f"No macro named '{name}'. Say 'list macros' to see what's available."
    results = execute_macro(macro)
    summary = []
    for tool, result in results:
        summary.append(f"{tool}: {result[:100]}")
    return f"Ran macro '{macro['name']}' ({len(results)} steps):\n" + "\n".join(summary)


@_register("add_macro")
def _add_macro_tool(name: str = "", steps: list | None = None, _raw: str = "", **kwargs) -> str:
    """Add a macro. Expects structured steps or parses from raw text."""
    from vox.macros import add_macro
    if not name and _raw:
        # Try to parse: "create macro morning briefing: weather, notes"
        m = re.search(
            r"(?:macro|routine)\s+['\"]?(.+?)['\"]?\s*:\s*(.+)",
            _raw, re.IGNORECASE,
        )
        if m:
            name = m.group(1).strip()
            tool_list = [t.strip() for t in m.group(2).split(",")]
            steps = [{"tool": t, "args": {}} for t in tool_list if t]
    if not name:
        return "I need a macro name and steps. Try: 'create macro morning briefing: get_weather, list_notes'"
    if not steps:
        return f"I need steps for macro '{name}'. Format: 'create macro {name}: tool1, tool2'"
    add_macro(name, steps)
    step_names = [s.get("tool", "?") for s in steps]
    return f"Macro '{name}' saved with {len(steps)} steps: {' → '.join(step_names)}"


@_register("list_macros")
def _list_macros_tool(**kwargs) -> str:
    """List all saved macros."""
    from vox.macros import list_macros
    macros = list_macros()
    if not macros:
        return "No macros yet. Say 'create macro name: tool1, tool2' to add one."
    lines = []
    for m in macros:
        steps = [s.get("tool", "?") for s in m.get("steps", [])]
        lines.append(f"'{m['name']}': {' → '.join(steps)}")
    return f"{len(macros)} macros:\n" + "\n".join(lines)


@_register("remove_macro")
def _remove_macro(name: str = "", **kwargs) -> str:
    """Remove a macro."""
    from vox.macros import remove_macro
    if not name:
        return "Which macro should I remove?"
    if remove_macro(name):
        return f"Macro '{name}' removed."
    return f"No macro named '{name}'."


# ---------------------------------------------------------------------------
# Reminders — time-based, persistent (JSON-backed)
# ---------------------------------------------------------------------------

_REMINDERS_FILE = None


def _get_reminders_file():
    global _REMINDERS_FILE
    if _REMINDERS_FILE is None:
        from vox.config import PROJECT_ROOT
        data_dir = PROJECT_ROOT / "data"
        data_dir.mkdir(exist_ok=True)
        _REMINDERS_FILE = data_dir / "reminders.json"
    return _REMINDERS_FILE


def _load_reminders() -> list[dict]:
    import json
    path = _get_reminders_file()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def _save_reminders(reminders: list[dict]):
    import json
    path = _get_reminders_file()
    path.write_text(json.dumps(reminders, indent=2, default=str), encoding="utf-8")


def _parse_time_offset(text: str) -> int:
    """Parse a time offset from text, returning minutes. 0 if unparseable."""
    m = re.search(r"(\d+)\s*(minutes?|mins?|m)\b", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*(hours?|hrs?|h)\b", text, re.IGNORECASE)
    if m:
        return int(m.group(1)) * 60
    m = re.search(r"(\d+)\s*(seconds?|secs?|s)\b", text, re.IGNORECASE)
    if m:
        return max(1, int(m.group(1)) // 60)
    return 0


def check_reminders() -> list[str]:
    """Check for due reminders. Called on each request."""
    reminders = _load_reminders()
    now = datetime.datetime.now()
    due = []
    remaining = []
    for r in reminders:
        due_at = datetime.datetime.fromisoformat(r["due_at"])
        if now >= due_at:
            due.append(f"Reminder: {r['text']}")
        else:
            remaining.append(r)
    if due:
        _save_reminders(remaining)
    return due


@_register("set_reminder")
def _set_reminder(text: str = "", minutes: int = 0, _raw: str = "", **kwargs) -> str:
    """Set a reminder."""
    if _raw and not text:
        # Parse: "remind me in 30 minutes to check the oven"
        cleaned = re.sub(
            r"^(remind\s+me|set\s+a?\s*(reminder|timer|alarm))\s*",
            "", _raw, flags=re.IGNORECASE,
        ).strip()
        minutes = minutes or _parse_time_offset(cleaned)
        text = re.sub(
            r"\b(in\s+)?\d+\s*(minute|min|hour|hr|second|sec|m|h|s)\w*\s*(to\s+)?",
            "", cleaned, flags=re.IGNORECASE,
        ).strip()
    if not text:
        return "What should I remind you about?"
    if not minutes:
        minutes = 30  # default 30 min

    due_at = datetime.datetime.now() + datetime.timedelta(minutes=minutes)
    reminders = _load_reminders()
    reminder = {
        "id": len(reminders) + 1,
        "text": text,
        "due_at": due_at.isoformat(),
        "created": datetime.datetime.now().isoformat(),
    }
    reminders.append(reminder)
    _save_reminders(reminders)

    if minutes >= 60:
        time_str = f"{minutes // 60} hour(s) and {minutes % 60} minute(s)"
    else:
        time_str = f"{minutes} minute(s)"
    return f"Got it! I'll remind you in {time_str}: {text}"


@_register("list_reminders")
def _list_reminders(**kwargs) -> str:
    """List pending reminders."""
    reminders = _load_reminders()
    if not reminders:
        return "No pending reminders."
    now = datetime.datetime.now()
    lines = []
    for r in reminders:
        due_at = datetime.datetime.fromisoformat(r["due_at"])
        delta = due_at - now
        if delta.total_seconds() > 0:
            mins = int(delta.total_seconds() / 60)
            if mins >= 60:
                time_str = f"in {mins // 60}h {mins % 60}m"
            else:
                time_str = f"in {mins}m"
        else:
            time_str = "NOW"
        lines.append(f"#{r['id']} {r['text']} — due {time_str}")
    return f"{len(reminders)} reminder(s):\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# File navigator — safe directory listing and file search
# ---------------------------------------------------------------------------

_SAFE_DIRS = {
    "downloads": "Downloads",
    "documents": "Documents",
    "desktop": "Desktop",
}


def _resolve_safe_dir(name: str):
    """Resolve a safe directory name to a Path."""
    from pathlib import Path as _Path
    folder = _SAFE_DIRS.get(name.lower())
    if not folder:
        return None
    home = _Path.home()
    path = home / folder
    if not path.exists():
        # Try vox project downloads
        from vox.config import DOWNLOADS_DIR
        if name.lower() == "downloads":
            return DOWNLOADS_DIR
    return path if path.exists() else None


@_register("list_files")
def _list_files(directory: str = "downloads", sort: str = "newest", **kwargs) -> str:
    """List files in a safe directory."""
    dir_path = _resolve_safe_dir(directory)
    if not dir_path:
        return f"Unknown directory '{directory}'. Try: downloads, documents, or desktop."

    files = [f for f in dir_path.iterdir() if f.is_file()]
    if not files:
        return f"No files in {directory}."

    # Sort
    if sort == "newest":
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    elif sort == "oldest":
        files.sort(key=lambda f: f.stat().st_mtime)
    elif sort == "largest":
        files.sort(key=lambda f: f.stat().st_size, reverse=True)
    else:
        files.sort(key=lambda f: f.name.lower())

    # Format (limit to 20)
    lines = []
    for f in files[:20]:
        import datetime as _dt
        size = f.stat().st_size
        mtime = _dt.datetime.fromtimestamp(f.stat().st_mtime)
        if size > 1024 * 1024:
            size_str = f"{size / 1024 / 1024:.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} B"
        lines.append(f"{f.name} ({size_str}, {mtime:%Y-%m-%d %H:%M})")

    header = f"{len(files)} files in {directory}"
    if len(files) > 20:
        header += f" (showing 20 of {len(files)})"
    return f"{header}:\n" + "\n".join(lines)


@_register("find_file")
def _find_file(pattern: str = "", **kwargs) -> str:
    """Search for a file by name pattern in safe directories."""
    if not pattern:
        return "What file are you looking for?"
    import fnmatch

    results = []
    for dir_name in _SAFE_DIRS:
        dir_path = _resolve_safe_dir(dir_name)
        if not dir_path:
            continue
        for f in dir_path.iterdir():
            if f.is_file() and fnmatch.fnmatch(f.name.lower(), f"*{pattern.lower()}*"):
                results.append((dir_name, f))

    if not results:
        return f"No files matching '{pattern}' found in downloads/documents/desktop."

    lines = []
    for dir_name, f in results[:15]:
        import datetime as _dt
        mtime = _dt.datetime.fromtimestamp(f.stat().st_mtime)
        lines.append(f"[{dir_name}] {f.name} ({mtime:%Y-%m-%d %H:%M})")

    return f"Found {len(results)} file(s):\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Daily briefing — aggregates multiple tools into one response
# ---------------------------------------------------------------------------

@_register("daily_briefing")
def _daily_briefing(**kwargs) -> str:
    """Run a morning briefing: time, weather, notes, reminders."""
    sections = []

    # Time
    time_result = execute_tool("get_current_time", {})
    sections.append(f"It's {time_result}.")

    # Weather (if available)
    try:
        weather = execute_tool("get_weather", {})
        if weather and "error" not in weather.lower():
            sections.append(f"Weather: {weather}")
    except Exception:
        pass

    # Notes/to-do
    try:
        notes = execute_tool("list_notes", {})
        if notes and "no notes" not in notes.lower():
            # Count pending tasks
            pending = notes.count("[TODO]")
            sections.append(f"You have {pending} pending task(s)." if pending else notes)
    except Exception:
        pass

    if len(sections) <= 1:
        sections.append("That's all I have for now.")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# News / RSS feeds — parse with stdlib xml, no extra dependency
# ---------------------------------------------------------------------------

_RSS_FEEDS: dict[str, list[str]] = {
    "general": [
        "https://feeds.bbci.co.uk/news/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    ],
    "tech": [
        "https://feeds.arstechnica.com/arstechnica/index",
        "https://www.theverge.com/rss/index.xml",
    ],
    "science": [
        "https://rss.nytimes.com/services/xml/rss/nyt/Science.xml",
    ],
    "world": [
        "https://feeds.bbci.co.uk/news/world/rss.xml",
    ],
    "business": [
        "https://feeds.bbci.co.uk/news/business/rss.xml",
    ],
}


def _parse_rss(xml_text: str, max_items: int = 5) -> list[dict]:
    """Parse RSS XML into a list of {title, link, description} dicts."""
    import xml.etree.ElementTree as ET

    items = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    # Standard RSS 2.0
    for item in root.iter("item"):
        title = item.findtext("title", "").strip()
        link = item.findtext("link", "").strip()
        desc = item.findtext("description", "").strip()
        if title:
            # Strip HTML tags from description
            desc = re.sub(r"<[^>]+>", "", desc)[:200]
            items.append({"title": title, "link": link, "description": desc})
            if len(items) >= max_items:
                break

    # Atom feeds (entry instead of item)
    if not items:
        for ns in ["", "{http://www.w3.org/2005/Atom}"]:
            for entry in root.iter(f"{ns}entry"):
                title = (entry.findtext(f"{ns}title", "") or "").strip()
                link_el = entry.find(f"{ns}link")
                link = (link_el.get("href", "") if link_el is not None else "").strip()
                summary = (entry.findtext(f"{ns}summary", "") or "").strip()
                summary = re.sub(r"<[^>]+>", "", summary)[:200]
                if title:
                    items.append({"title": title, "link": link, "description": summary})
                    if len(items) >= max_items:
                        break
            if items:
                break

    return items


@_register("get_news")
def _get_news(topic: str = "general", **kwargs) -> str:
    """Fetch news headlines from RSS feeds."""
    import urllib.request
    import urllib.error

    topic = topic.lower().strip()
    feeds = _RSS_FEEDS.get(topic, _RSS_FEEDS["general"])

    all_items: list[dict] = []
    for url in feeds:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "VOX/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                xml_text = resp.read().decode("utf-8", errors="replace")
            all_items.extend(_parse_rss(xml_text, max_items=5))
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            log.warning("Failed to fetch RSS feed %s: %s", url, e)
            continue

    if not all_items:
        return f"Couldn't fetch any {topic} news right now. Check your internet connection."

    # Deduplicate by title
    seen: set[str] = set()
    unique = []
    for item in all_items:
        key = item["title"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)

    # Format top 8 headlines
    lines = []
    for i, item in enumerate(unique[:8], 1):
        line = f"{i}. {item['title']}"
        if item["description"]:
            line += f" — {item['description']}"
        lines.append(line)

    return f"Top {topic} headlines:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Media control — send media keys via PowerShell (Windows) or dbus (Linux)
# ---------------------------------------------------------------------------

_MEDIA_KEYS: dict[str, str] = {
    "play": "0xB3",       # VK_MEDIA_PLAY_PAUSE
    "pause": "0xB3",      # VK_MEDIA_PLAY_PAUSE (toggle)
    "next": "0xB0",       # VK_MEDIA_NEXT_TRACK
    "previous": "0xB1",   # VK_MEDIA_PREV_TRACK
    "volume_up": "0xAF",  # VK_VOLUME_UP
    "volume_down": "0xAE",  # VK_VOLUME_DOWN
    "mute": "0xAD",       # VK_VOLUME_MUTE
    "unmute": "0xAD",     # VK_VOLUME_MUTE (toggle)
}


def _send_media_key(vk_code: str) -> bool:
    """Send a virtual media key press on Windows via PowerShell."""
    import subprocess as _sp
    ps_script = (
        f'Add-Type -TypeDefinition @"\n'
        f'using System; using System.Runtime.InteropServices;\n'
        f'public class VKey {{\n'
        f'  [DllImport("user32.dll")] public static extern void keybd_event(byte bVk, byte bScan, uint dwFlags, UIntPtr dwExtraInfo);\n'
        f'}}\n'
        f'"@\n'
        f'[VKey]::keybd_event({vk_code}, 0, 0, [UIntPtr]::Zero)\n'
        f'Start-Sleep -Milliseconds 50\n'
        f'[VKey]::keybd_event({vk_code}, 0, 2, [UIntPtr]::Zero)'
    )
    try:
        result = _sp.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _get_now_playing() -> str:
    """Get currently playing media info on Windows."""
    import subprocess as _sp
    ps_script = (
        'Get-Process | Where-Object {$_.MainWindowTitle -ne ""} | '
        'Where-Object {$_.ProcessName -match "spotify|vlc|foobar|musicbee|winamp|groove|media"} | '
        'Select-Object -First 1 -ExpandProperty MainWindowTitle'
    )
    try:
        result = _sp.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True, text=True, timeout=5,
        )
        title = result.stdout.strip()
        if title:
            return title
        return "Couldn't detect what's playing — no music app window found."
    except Exception:
        return "Couldn't check — error querying media players."


@_register("media_control")
def _media_control(action: str = "play", **kwargs) -> str:
    """Control media playback via system media keys."""
    action = action.lower().strip()

    if action == "now_playing":
        return _get_now_playing()

    vk_code = _MEDIA_KEYS.get(action)
    if not vk_code:
        return f"Unknown media action: {action}. Try: play, pause, next, previous, volume_up, volume_down, mute."

    if sys.platform == "win32":
        ok = _send_media_key(vk_code)
        if ok:
            labels = {
                "play": "Playing!", "pause": "Paused.",
                "next": "Skipped to next track.", "previous": "Back to previous track.",
                "volume_up": "Volume up.", "volume_down": "Volume down.",
                "mute": "Muted.", "unmute": "Unmuted.",
            }
            return labels.get(action, f"Done — {action}.")
        return f"Failed to send media key for {action}."
    else:
        return "Media control currently only supported on Windows."


# ---------------------------------------------------------------------------
# Smart home — Home Assistant REST API
# ---------------------------------------------------------------------------

# Entity cache — fetched once per session, refreshed on demand
_hass_entities: list[dict] | None = None


def _hass_api(method: str, endpoint: str, json_data: dict | None = None) -> dict | str:
    """Make a Home Assistant REST API call."""
    import json
    import urllib.request

    from vox.config import HASS_TOKEN, HASS_URL

    if not HASS_URL or not HASS_TOKEN:
        return "Home Assistant not configured. Set HASS_URL and HASS_TOKEN in .env."

    url = f"{HASS_URL.rstrip('/')}/api/{endpoint}"
    headers = {
        "Authorization": f"Bearer {HASS_TOKEN}",
        "Content-Type": "application/json",
    }
    data = json.dumps(json_data).encode() if json_data else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
        return json.loads(resp.read())


def _hass_get_entities() -> list[dict]:
    """Get all HA entities (cached)."""
    global _hass_entities
    if _hass_entities is None:
        result = _hass_api("GET", "states")
        if isinstance(result, list):
            _hass_entities = result
        else:
            return []
    return _hass_entities


def _hass_find_entity(hint: str, domain: str | None = None) -> str | None:
    """Fuzzy-match an entity_id from a natural language hint."""
    from difflib import SequenceMatcher

    entities = _hass_get_entities()
    hint_lower = hint.lower().replace(" ", "_")

    best_id, best_score = None, 0.0
    for e in entities:
        eid = e.get("entity_id", "")
        if domain and not eid.startswith(f"{domain}."):
            continue
        friendly = e.get("attributes", {}).get("friendly_name", "").lower().replace(" ", "_")
        # Score against both entity_id and friendly_name
        for candidate in [eid.split(".", 1)[-1] if "." in eid else eid, friendly]:
            if hint_lower in candidate:
                score = 0.95
            else:
                score = SequenceMatcher(None, hint_lower, candidate).ratio()
            if score > best_score:
                best_score = score
                best_id = eid

    return best_id if best_score > 0.4 else None


@_register("smart_home")
def _smart_home(
    action: str = "",
    entity_hint: str = "",
    state: str = "",
    temperature: int = 0,
    scene_name: str = "",
    **kwargs,
) -> str:
    """Control Home Assistant devices."""
    from vox.config import HASS_TOKEN, HASS_URL

    if not HASS_URL or not HASS_TOKEN:
        return "Home Assistant not configured. Add HASS_URL and HASS_TOKEN to your .env file."

    try:
        if action == "toggle":
            domain = _guess_domain(entity_hint)
            entity_id = _hass_find_entity(entity_hint, domain)
            if not entity_id:
                return f"Couldn't find a device matching '{entity_hint}'."
            service = f"turn_{state}" if state in ("on", "off") else "toggle"
            _hass_api("POST", f"services/{domain}/{service}", {"entity_id": entity_id})
            return f"{'Turned' if state else 'Toggled'} {state} {entity_hint}."

        elif action == "climate":
            entity_id = _hass_find_entity("thermostat", "climate") or "climate.thermostat"
            _hass_api("POST", "services/climate/set_temperature", {
                "entity_id": entity_id,
                "temperature": temperature,
            })
            return f"Thermostat set to {temperature}°."

        elif action in ("lock", "unlock"):
            entity_id = _hass_find_entity(entity_hint, "lock")
            if not entity_id:
                return f"Couldn't find a lock matching '{entity_hint}'."
            _hass_api("POST", f"services/lock/{action}", {"entity_id": entity_id})
            return f"{'Locked' if action == 'lock' else 'Unlocked'} {entity_hint}."

        elif action == "status":
            entity_id = _hass_find_entity(entity_hint)
            if not entity_id:
                return f"Couldn't find a device matching '{entity_hint}'."
            states = _hass_api("GET", f"states/{entity_id}")
            if isinstance(states, dict):
                s = states.get("state", "unknown")
                name = states.get("attributes", {}).get("friendly_name", entity_id)
                return f"{name} is {s}."
            return f"Couldn't get status for {entity_hint}."

        elif action == "scene":
            entity_id = _hass_find_entity(scene_name, "scene")
            if not entity_id:
                return f"Couldn't find scene '{scene_name}'."
            _hass_api("POST", "services/scene/turn_on", {"entity_id": entity_id})
            return f"Activated scene: {scene_name}."

        else:
            return f"Unknown smart home action: {action}"

    except Exception as e:
        return f"Home Assistant error: {e}"


def _guess_domain(hint: str) -> str:
    """Guess HA entity domain from a natural language hint."""
    h = hint.lower()
    if any(w in h for w in ("light", "lamp", "bulb")):
        return "light"
    if any(w in h for w in ("fan", "ceiling")):
        return "fan"
    if any(w in h for w in ("switch", "plug", "outlet")):
        return "switch"
    if any(w in h for w in ("blind", "curtain", "shade")):
        return "cover"
    return "light"  # default to light


# ---------------------------------------------------------------------------
# RAG — document search and indexing
# ---------------------------------------------------------------------------

@_register("search_documents")
def _search_documents(query: str = "", **kwargs) -> str:
    """Search indexed documents for relevant information."""
    if not query:
        return "What would you like to search for in your documents?"

    try:
        from vox.rag import search
        hits = search(query, n_results=5)
    except ImportError:
        return "Document search not available. Install with: pip install -e '.[memory]'"
    except Exception as e:
        return f"Document search failed: {e}"

    if not hits:
        return f"No relevant documents found for: {query}. Try indexing with 'index my documents'."

    lines = []
    for i, hit in enumerate(hits, 1):
        source = hit["source"]
        score = hit["score"]
        text = hit["text"][:300]
        lines.append(f"{i}. [{source}] (relevance: {score:.0%})\n   {text}")

    return f"Found {len(hits)} relevant section(s):\n\n" + "\n\n".join(lines)


@_register("index_documents")
def _index_documents(directory: str = "", **kwargs) -> str:
    """Index local documents for search."""
    try:
        from vox.rag import get_stats, index_directory
    except ImportError:
        return "Document indexing not available. Install with: pip install -e '.[memory]'"

    from pathlib import Path
    dir_path = Path(directory) if directory else None

    try:
        chunks = index_directory(dir_path)
        stats = get_stats()
        return (
            f"Indexed {chunks} new chunks. "
            f"Total: {stats['chunks']} chunks from {stats['files']} files."
        )
    except Exception as e:
        return f"Indexing failed: {e}"


# ---------------------------------------------------------------------------
# Image upscaling
# ---------------------------------------------------------------------------

@_register("upscale_image")
def _upscale_image(path: str = "", scale: int = 2, **kwargs) -> str:
    """Upscale an image using Real-ESRGAN (GPU) or PIL Lanczos (CPU fallback)."""
    if not path:
        return "No image path provided. Tell me which image to upscale."
    from pathlib import Path as _Path

    img_path = _Path(path)
    if not img_path.exists():
        from vox.config import DOWNLOADS_DIR
        img_path = DOWNLOADS_DIR / path
    if not img_path.exists():
        return f"Image not found: {path}"

    scale = max(2, min(scale, 4))
    out_name = f"{img_path.stem}_upscaled_{scale}x{img_path.suffix}"
    from vox.config import DOWNLOADS_DIR
    out_path = DOWNLOADS_DIR / out_name

    # Try Real-ESRGAN first (GPU, high quality)
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        import torch
        import numpy as np
        from PIL import Image

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4,
        )
        upsampler = RealESRGANer(
            scale=4, model_path=None, model=model,
            tile=0, tile_pad=10, pre_pad=0, half=True,
        )
        img = np.array(Image.open(img_path).convert("RGB"))
        output, _ = upsampler.enhance(img, outscale=scale)
        Image.fromarray(output).save(str(out_path))
        w, h = output.shape[1], output.shape[0]
        log.info("Real-ESRGAN upscaled %s → %s (%dx%d)", path, out_path, w, h)
        return f"Upscaled {scale}x with Real-ESRGAN! {w}x{h} saved to {out_name}"
    except ImportError:
        log.info("Real-ESRGAN not installed, using PIL Lanczos fallback")
    except Exception as e:
        log.warning("Real-ESRGAN failed: %s, falling back to PIL", e)

    # Fallback: PIL Lanczos (CPU, decent quality)
    try:
        from PIL import Image
        img = Image.open(img_path)
        new_size = (img.width * scale, img.height * scale)
        upscaled = img.resize(new_size, Image.LANCZOS)
        upscaled.save(str(out_path))
        log.info("PIL upscaled %s → %s (%dx%d)", path, out_path, *new_size)
        return (f"Upscaled {scale}x with Lanczos! "
                f"{new_size[0]}x{new_size[1]} saved to {out_name}")
    except Exception as e:
        return f"Upscaling failed: {e}"


# ---------------------------------------------------------------------------
# Code runner — safe Python expression/code execution
# ---------------------------------------------------------------------------

@_register("run_code")
def _run_code(expression: str = "", code: str = "", **kwargs) -> str:
    """Run a Python expression or code snippet in a subprocess sandbox."""
    import subprocess as _sp
    snippet = code or expression
    if not snippet:
        return "No code or expression provided."

    # For simple math expressions, try eval first (fast path)
    if not code and expression:
        expr = expression.replace("\u00d7", "*").replace("\u00f7", "/")
        expr = re.sub(r"\b(plus|and)\b", "+", expr, flags=re.IGNORECASE)
        expr = re.sub(r"\bminus\b", "-", expr, flags=re.IGNORECASE)
        expr = re.sub(r"\b(times|multiplied\s+by)\b", "*", expr, flags=re.IGNORECASE)
        expr = re.sub(r"\b(divided\s+by|over)\b", "/", expr, flags=re.IGNORECASE)
        expr = re.sub(r"\bto\s+the\s+power\s+of\b", "**", expr, flags=re.IGNORECASE)
        expr = re.sub(r"\bsquared\b", "**2", expr, flags=re.IGNORECASE)
        expr = re.sub(r"\bcubed\b", "**3", expr, flags=re.IGNORECASE)
        expr = re.sub(r"\bsqrt\b", "math.sqrt", expr, flags=re.IGNORECASE)
        expr = re.sub(r"[%$,]", "", expr).strip()
        # Temperature conversion shortcuts
        temp = re.search(
            r"([\d.]+)\s*\u00b0?\s*([FCfc])\s+(?:to|in)\s+\u00b0?\s*([FCfc])",
            expr,
        )
        if temp:
            val = float(temp.group(1))
            fr, to = temp.group(2).upper(), temp.group(3).upper()
            if fr == "F" and to == "C":
                return f"{val}\u00b0F = {(val - 32) * 5 / 9:.1f}\u00b0C"
            if fr == "C" and to == "F":
                return f"{val}\u00b0C = {val * 9 / 5 + 32:.1f}\u00b0F"
        snippet = f"import math; print({expr})"

    # Run in subprocess for safety
    try:
        result = _sp.run(
            [sys.executable, "-c", snippet],
            capture_output=True, text=True, timeout=10,
        )
        output = result.stdout.strip()
        if result.stderr.strip():
            output += f"\nError: {result.stderr.strip()[:500]}"
        if not output:
            output = "(no output)"
        return output[:3000]
    except _sp.TimeoutExpired:
        return "Code timed out after 10 seconds."
    except Exception as e:
        return f"Code execution failed: {e}"


# ---------------------------------------------------------------------------
# Clipboard — read/write system clipboard (no extra dependencies)
# ---------------------------------------------------------------------------

@_register("read_clipboard")
def _read_clipboard(**kwargs) -> str:
    """Read text from the system clipboard."""
    import subprocess
    import sys
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                ["powershell", "-Command", "Get-Clipboard"],
                capture_output=True, text=True, timeout=5,
            )
            text = result.stdout.strip()
        else:
            result = subprocess.run(
                ["xclip", "-selection", "clipboard", "-o"],
                capture_output=True, text=True, timeout=5,
            )
            text = result.stdout.strip()
        if not text:
            return "Clipboard is empty."
        return f"Clipboard contents:\n{text[:3000]}"
    except Exception as e:
        return f"Couldn't read clipboard: {e}"


@_register("write_clipboard")
def _write_clipboard(text: str = "", **kwargs) -> str:
    """Write text to the system clipboard."""
    import subprocess
    import sys
    if not text:
        return "Nothing to copy — no text provided."
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["clip"],
                input=text, text=True, timeout=5,
            )
        else:
            subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=text, text=True, timeout=5,
            )
        return f"Copied to clipboard! ({len(text)} characters)"
    except Exception as e:
        return f"Couldn't write to clipboard: {e}"


# ---------------------------------------------------------------------------
# Safe system commands — strict whitelist, no arbitrary execution
# ---------------------------------------------------------------------------

_SAFE_COMMANDS = {
    "disk_space": {
        "win32": [
            "powershell", "-Command",
            "Get-PSDrive -PSProvider FileSystem |"
            " Format-Table Name,Used,Free -AutoSize",
        ],
        "linux": ["df", "-h"],
        "desc": "Check disk space",
    },
    "gpu_processes": {
        "win32": ["nvidia-smi"],
        "linux": ["nvidia-smi"],
        "desc": "GPU usage and processes",
    },
    "running_processes": {
        "win32": ["tasklist", "/FO", "CSV", "/NH"],
        "linux": ["ps", "aux", "--sort=-pcpu"],
        "desc": "Running processes",
    },
    "open_chrome": {
        "win32": ["start", "chrome"],
        "linux": ["google-chrome"],
        "desc": "Open Chrome browser",
    },
    "restart_ollama": {
        "win32": ["cmd", "/c", "taskkill /F /IM ollama.exe & start ollama serve"],
        "linux": ["systemctl", "restart", "ollama"],
        "desc": "Restart Ollama LLM server",
    },
    "network_status": {
        "win32": ["ipconfig"],
        "linux": ["ip", "addr"],
        "desc": "Network configuration",
    },
}


@_register("run_command")
def _run_command(command: str = "", **kwargs) -> str:
    """Execute a whitelisted system command."""
    import platform
    import subprocess

    if not command:
        available = ", ".join(f"{k} ({v['desc']})" for k, v in _SAFE_COMMANDS.items())
        return f"Available commands: {available}"

    cmd_key = command.lower().replace(" ", "_").replace("-", "_")
    cmd_def = _SAFE_COMMANDS.get(cmd_key)
    if cmd_def is None:
        available = ", ".join(_SAFE_COMMANDS.keys())
        return f"Unknown command '{command}'. Available: {available}"

    os_key = "win32" if platform.system() == "Windows" else "linux"
    cmd_args = cmd_def.get(os_key)
    if not cmd_args:
        return f"Command '{command}' not supported on {platform.system()}"

    try:
        result = subprocess.run(
            cmd_args,
            capture_output=True,
            text=True,
            timeout=15,
            shell=(os_key == "win32" and cmd_args[0] in ("start", "cmd")),
        )
        output = result.stdout.strip()
        if result.returncode != 0 and result.stderr:
            output += f"\nError: {result.stderr.strip()}"
        if len(output) > 3000:
            output = output[:3000] + "\n... (truncated)"
        return output or "Command completed (no output)."
    except subprocess.TimeoutExpired:
        return f"Command '{command}' timed out after 15 seconds."
    except Exception as e:
        return f"Command failed: {e}"


@_register("take_screenshot")
def _take_screenshot(**kwargs) -> str:
    """Capture a screenshot and save to downloads."""
    try:
        from PIL import ImageGrab
    except ImportError:
        return "Screenshot requires Pillow. Install with: pip install Pillow"

    from vox.config import DOWNLOADS_DIR

    try:
        img = ImageGrab.grab()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vox_screenshot_{timestamp}.png"
        filepath = DOWNLOADS_DIR / filename
        img.save(filepath)
        log.info("Screenshot saved: %s (%dx%d)", filepath, img.width, img.height)
        return f"Screenshot saved to {filename} ({img.width}x{img.height})"
    except Exception as e:
        return f"Screenshot failed: {e}"


def _clean_ddg_url(url: str) -> str:
    """Extract the real URL from a DuckDuckGo redirect wrapper.

    DDG HTML lite wraps links like:
        //duckduckgo.com/l/?uddg=https%3A%2F%2Freal-url.com&rut=abc123
    This extracts and returns the decoded ``uddg`` parameter value.
    Non-redirect URLs are returned unchanged.
    """
    if not url:
        return url
    import urllib.parse as _urlparse

    parsed = _urlparse.urlparse(url)
    if parsed.hostname and "duckduckgo.com" in parsed.hostname and parsed.path.startswith("/l/"):
        qs = _urlparse.parse_qs(parsed.query)
        uddg = qs.get("uddg")
        if uddg:
            return uddg[0]
    return url


_ANON_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) "
                   "Gecko/20100101 Firefox/128.0"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "DNT": "1",
    "Sec-GPC": "1",
}


def _search_brave(query: str) -> tuple[list[str], list[str]]:
    """Search via Brave Search API. Returns (results, urls)."""
    import json
    import urllib.parse
    import urllib.request
    from vox.config import BRAVE_API_KEY

    if not BRAVE_API_KEY:
        return [], []
    encoded = urllib.parse.urlencode({"q": query, "count": 8})
    url = f"https://api.search.brave.com/res/v1/web/search?{encoded}"
    req = urllib.request.Request(url, headers={
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY,
    })
    with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
        data = json.loads(resp.read())

    results, urls = [], []
    for item in data.get("web", {}).get("results", [])[:8]:
        title = item.get("title", "")
        snippet = item.get("description", "")
        link = item.get("url", "")
        results.append(f"- {title}: {snippet}")
        if link:
            results.append(f"  Link: {link}")
            urls.append(link)
    return results, urls


def _search_searxng(query: str) -> tuple[list[str], list[str]]:
    """Search via self-hosted SearXNG instance. Returns (results, urls)."""
    import json
    import urllib.parse
    import urllib.request
    from vox.config import SEARXNG_URL

    if not SEARXNG_URL:
        return [], []
    encoded = urllib.parse.urlencode({"q": query, "format": "json", "categories": "general"})
    url = f"{SEARXNG_URL.rstrip('/')}/search?{encoded}"
    req = urllib.request.Request(url, headers=_ANON_HEADERS)
    with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
        data = json.loads(resp.read())

    results, urls = [], []
    for item in data.get("results", [])[:8]:
        title = item.get("title", "")
        snippet = item.get("content", "")
        link = item.get("url", "")
        results.append(f"- {title}: {snippet}")
        if link:
            results.append(f"  Link: {link}")
            urls.append(link)
    return results, urls


def _search_ddg(query: str) -> tuple[list[str], list[str]]:
    """Search via DuckDuckGo (no API key). Returns (results, urls)."""
    import json
    import urllib.parse
    import urllib.request

    encoded = urllib.parse.urlencode({"q": query, "format": "json"})
    url = f"https://api.duckduckgo.com/?{encoded}"
    req = urllib.request.Request(url, headers=_ANON_HEADERS)
    with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
        data = json.loads(resp.read())

    results, urls = [], []
    if data.get("Abstract"):
        results.append(f"Summary: {data['Abstract']}")
        if data.get("AbstractURL"):
            results.append(f"Source: {data['AbstractURL']}")
            urls.append(data["AbstractURL"])

    for topic in data.get("RelatedTopics", [])[:5]:
        if isinstance(topic, dict) and topic.get("Text"):
            results.append(f"- {topic['Text']}")
            topic_url = topic.get("FirstURL", "")
            if topic_url:
                results.append(f"  Link: {topic_url}")
                urls.append(topic_url)

    if not results:
        # Fallback: scrape DuckDuckGo HTML lite
        html_url = f"https://html.duckduckgo.com/html/?{urllib.parse.urlencode({'q': query})}"
        req = urllib.request.Request(html_url, headers=_ANON_HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            html = resp.read().decode("utf-8", errors="replace")

        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL)
        links = re.findall(r'class="result__a"[^>]*href="([^"]*)"', html)
        if not links:
            links = re.findall(r'class="result__url"[^>]*href="([^"]*)"', html)
        for i, snippet in enumerate(snippets[:8]):
            clean = re.sub(r"<[^>]+>", "", snippet).strip()
            link = _clean_ddg_url(links[i]) if i < len(links) else ""
            results.append(f"- {clean}")
            if link:
                results.append(f"  Link: {link}")
                urls.append(link)

    return results, urls


@_register("web_search")
def _web_search(query: str = "", deep: bool = False, **kwargs) -> str:
    """Search the web. Backend configurable via SEARCH_ENGINE env var.

    Supports: brave (Brave Search API), searxng (self-hosted), ddg (DuckDuckGo, default).
    When deep=True, fetches and extracts content from top result URLs.
    """
    if not query:
        return "No search query provided."

    from vox.config import SEARCH_ENGINE

    # Try configured backend, fall back to DDG
    backends = {
        "brave": _search_brave,
        "searxng": _search_searxng,
        "ddg": _search_ddg,
    }

    results, result_urls = [], []
    engine = SEARCH_ENGINE.lower().strip()
    search_fn = backends.get(engine, _search_ddg)

    try:
        results, result_urls = search_fn(query)
    except Exception as e:
        log.warning("Search backend %s failed: %s — falling back to DDG", engine, e)
        if engine != "ddg":
            try:
                results, result_urls = _search_ddg(query)
            except Exception as e2:
                return f"Search failed: {e2}"

    if not results:
        return f"No results found for: {query}"

    output = f"Search results for '{query}':\n" + "\n".join(results)

    # Deep search: fetch and extract content from top result pages
    if deep and result_urls:
        deep_content = _deep_fetch_results(result_urls[:5], _ANON_HEADERS)
        if deep_content:
            output += "\n\n--- Deep Search Results ---\n" + deep_content

    return output


def _deep_fetch_results(urls: list[str], headers: dict) -> str:
    """Fetch and extract text content from multiple URLs for deep search."""
    import urllib.request

    sections = []
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as resp:
                ct = resp.headers.get("Content-Type", "")
                if "text/html" not in ct and "text/plain" not in ct:
                    continue
                raw = resp.read().decode("utf-8", errors="replace")

            # Extract meaningful content
            text = re.sub(r"<script[^>]*>.*?</script>", "", raw, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
            text = re.sub(r"<nav[^>]*>.*?</nav>", "", text, flags=re.DOTALL)
            text = re.sub(r"<header[^>]*>.*?</header>", "", text, flags=re.DOTALL)
            text = re.sub(r"<footer[^>]*>.*?</footer>", "", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()

            if len(text) < 100:
                continue
            # Keep first 1500 chars per page to stay within context limits
            if len(text) > 1500:
                text = text[:1500] + "..."
            sections.append(f"[{url}]\n{text}")
        except Exception as e:
            log.debug("Deep fetch failed for %s: %s", url, e)
            continue

    if not sections:
        return ""
    return "\n\n".join(sections)


@_register("send_email")
def _send_email(
    to: str = "",
    subject: str = "",
    body: str = "",
    attachments: list[str] | str = "",
    **kwargs,
) -> str:
    """Send an email via SMTP, optionally with file attachments."""
    import mimetypes
    import os
    import smtplib
    from email import encoders
    from email.mime.base import MIMEBase
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    from vox.config import SMTP_FROM, SMTP_HOST, SMTP_PASSWORD, SMTP_PORT, SMTP_USER

    log.info("send_email called: to=%s, subject=%s, attachments=%s", to, subject, type(attachments).__name__)

    if not to:
        log.warning("send_email: no recipient provided")
        return "No recipient email address provided."
    if not SMTP_HOST:
        log.warning("send_email: SMTP_HOST not configured")
        return "Email not configured. Set SMTP_HOST in .env"

    log.info("SMTP config: host=%s, port=%s, from=%s, user=%s",
             SMTP_HOST, SMTP_PORT, SMTP_FROM, SMTP_USER or "(none)")

    # Normalize attachments to a list
    if isinstance(attachments, str):
        attachment_list = [attachments] if attachments else []
    else:
        attachment_list = list(attachments) if attachments else []

    warnings: list[str] = []

    try:
        if attachment_list:
            msg = MIMEMultipart()
            msg.attach(MIMEText(body))

            for filepath in attachment_list:
                # Security: restrict attachments to DOWNLOADS_DIR to prevent path traversal
                from vox.config import DOWNLOADS_DIR as _dl_dir
                resolved = os.path.realpath(filepath)
                downloads_real = os.path.realpath(str(_dl_dir))
                if not resolved.startswith(downloads_real):
                    warnings.append(f"Attachment outside downloads dir, blocked: {filepath}")
                    log.warning("BLOCKED attachment path traversal: %s → %s", filepath, resolved)
                    continue
                if not os.path.isfile(filepath):
                    warnings.append(f"Attachment not found, skipped: {filepath}")
                    continue

                content_type, _ = mimetypes.guess_type(filepath)
                if content_type is None:
                    content_type = "application/octet-stream"
                maintype, subtype = content_type.split("/", 1)

                with open(filepath, "rb") as f:
                    part = MIMEBase(maintype, subtype)
                    part.set_payload(f.read())

                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    "attachment",
                    filename=os.path.basename(filepath),
                )
                msg.attach(part)
        else:
            msg = MIMEText(body)

        msg["Subject"] = subject or "Message from VOX"
        msg["From"] = SMTP_FROM or SMTP_USER or f"vox@{SMTP_HOST}"
        msg["To"] = to

        log.info("Connecting to SMTP %s:%s ...", SMTP_HOST, SMTP_PORT)
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.ehlo()
            has_tls = server.has_extn("starttls")
            log.info("STARTTLS available: %s", has_tls)
            if has_tls:
                server.starttls()
                server.ehlo()
            if SMTP_USER and SMTP_PASSWORD:
                log.info("Authenticating as %s ...", SMTP_USER)
                server.login(SMTP_USER, SMTP_PASSWORD)
            log.info("Sending message: from=%s to=%s subject=%s", msg["From"], to, msg["Subject"])
            server.send_message(msg)
            log.info("SMTP send_message completed successfully")

        result = f"Email sent to {to} with subject: {subject}"
        if warnings:
            result += "\nWarnings: " + "; ".join(warnings)
        log.info("send_email result: %s", result)
        return result

    except Exception as e:
        log.exception("send_email failed: %s", e)
        return f"Failed to send email: {e}"


@_register("get_map")
def _get_map(location: str = "", map_type: str = "satellite", zoom: int = 18, **kwargs) -> str:
    """Fetch a real map/satellite image for a location using Google Maps Static API.

    Falls back to generating a clickable Google Maps link if no API key is set.
    """
    import urllib.parse
    import urllib.request

    from vox.config import DOWNLOADS_DIR, GOOGLE_MAPS_API_KEY

    if not location:
        return "No location provided."

    # Generate the Google Maps URL the user can always click
    maps_url = f"https://www.google.com/maps/search/{urllib.parse.quote(location)}/@?t=k"

    if not GOOGLE_MAPS_API_KEY:
        log.info("get_map: no API key, returning Google Maps link")
        return (
            f"Google Maps link for '{location}': {maps_url}\n"
            f"(Set GOOGLE_MAPS_API_KEY in .env for inline satellite images)"
        )

    # Fetch static map image from Google Maps API
    params = urllib.parse.urlencode({
        "center": location,
        "zoom": zoom,
        "size": "1024x1024",
        "maptype": map_type,  # satellite, roadmap, terrain, hybrid
        "key": GOOGLE_MAPS_API_KEY,
    })
    api_url = f"https://maps.googleapis.com/maps/api/staticmap?{params}"

    try:
        req = urllib.request.Request(api_url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            img_data = resp.read()
            content_type = resp.headers.get("Content-Type", "")

        if "image" not in content_type:
            log.warning("get_map: unexpected content type: %s", content_type)
            return f"Map API returned unexpected content. Google Maps link: {maps_url}"

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vox_map_{ts}.png"
        filepath = DOWNLOADS_DIR / filename
        filepath.write_bytes(img_data)
        log.info("Map image saved: %s (%d KB)", filepath, len(img_data) // 1024)
        return f"Map image saved to {filename}\nGoogle Maps: {maps_url}"

    except Exception as e:
        log.exception("get_map failed: %s", e)
        return f"Map fetch failed: {e}\nGoogle Maps link: {maps_url}"


@_register("web_fetch")
def _web_fetch(url: str = "", **kwargs) -> str:
    """Fetch a URL: return text for HTML, save file for PDFs."""
    import urllib.request

    from vox.config import DOWNLOADS_DIR

    if not url:
        return "No URL provided."
    if not re.match(r"https?://", url):
        return f"Invalid URL: {url}"

    # Security: block SSRF to internal/private networks
    from urllib.parse import urlparse
    _parsed = urlparse(url)
    _host = _parsed.hostname or ""
    _BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]", "metadata.google.internal"}
    if _host in _BLOCKED_HOSTS or _host.startswith(("10.", "192.168.", "169.254.")):
        return f"Blocked: cannot fetch internal/private URLs ({_host})"
    if _host.startswith("172.") and 16 <= int(_host.split(".")[1]) <= 31:
        return f"Blocked: cannot fetch internal/private URLs ({_host})"

    try:
        _fetch_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
            "DNT": "1",
            "Sec-GPC": "1",
        }
        req = urllib.request.Request(url, headers=_fetch_headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_type = resp.headers.get("Content-Type", "")
            data = resp.read()

        if "application/pdf" in content_type or url.lower().endswith(".pdf"):
            # Save PDF to downloads directory
            filename = url.split("/")[-1].split("?")[0]
            filename = re.sub(r"[^\w.\-]", "_", filename)
            if not filename.lower().endswith(".pdf"):
                filename += ".pdf"
            filepath = DOWNLOADS_DIR / filename
            filepath.write_bytes(data)
            size_kb = len(data) / 1024
            return f"PDF saved to {filepath} ({size_kb:.1f} KB)"
        else:
            # HTML or other text — strip tags and return text
            text = data.decode("utf-8", errors="replace")
            text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > 2000:
                text = text[:2000] + "..."
            return f"Content from {url}:\n{text}"

    except Exception as e:
        return f"Fetch failed: {e}"




def _should_use_nsfw_model(prompt: str, selfie: bool, nsfw_unlocked: bool = False) -> bool:
    """Determine if a request should route to the NSFW-capable model.

    NSFW is only enabled when the user includes the LoRA trigger word (easter egg).
    Without it, all output stays SFW regardless of what is requested.
    """
    from vox.config import IMAGE_NSFW_FILTER
    # If NSFW filter is on, never use NSFW model
    if IMAGE_NSFW_FILTER.lower() != "off":
        return False
    # Easter egg: trigger word required to unlock NSFW
    if not nsfw_unlocked:
        return False
    # Selfie/persona requests with unlock → NSFW model
    if selfie:
        return True
    # Check for NSFW keywords in the prompt
    return bool(_NSFW_KEYWORDS.search(prompt))


def _is_single_file_checkpoint(model_id: str) -> bool:
    """Check if a HuggingFace repo contains only a single-file checkpoint (no diffusers format)."""
    try:
        from huggingface_hub import model_info
        info = model_info(model_id)
        safetensor_files = [s for s in info.siblings if s.rfilename.endswith(".safetensors")]
        has_model_index = any(s.rfilename == "model_index.json" for s in info.siblings)
        # Single-file checkpoint: has .safetensors but no model_index.json (diffusers format)
        return bool(safetensor_files) and not has_model_index
    except Exception:
        return False


def _get_single_file_url(model_id: str) -> str | None:
    """Get the download URL for the single .safetensors file in a HuggingFace repo."""
    try:
        from huggingface_hub import model_info
        info = model_info(model_id)
        for s in info.siblings:
            if s.rfilename.endswith(".safetensors"):
                return f"https://huggingface.co/{model_id}/resolve/main/{s.rfilename}"
        return None
    except Exception:
        return None


# Model manager handles load-on-demand / unload-after-use for VRAM efficiency.
# Old permanent cache replaced — SDXL now unloads after generation, freeing ~12GB.
from vox import model_manager as _mm


def _load_pipeline(pipeline_cls, model_id: str, dtype):
    """Load a diffusers pipeline, auto-detecting single-file vs diffusers-format checkpoints."""
    if _is_single_file_checkpoint(model_id):
        url = _get_single_file_url(model_id)
        if url:
            log.info("Single-file checkpoint detected, using from_single_file: %s", url)
            from vox.config import IMAGE_NSFW_FILTER
            sf_kwargs = {"torch_dtype": dtype}
            if IMAGE_NSFW_FILTER.lower() == "off":
                sf_kwargs["safety_checker"] = None
                sf_kwargs["requires_safety_checker"] = False
            return pipeline_cls.from_single_file(url, **sf_kwargs)
    # Standard diffusers format
    pipe_kwargs = {"torch_dtype": dtype}
    # Disable safety checker when NSFW filter is off
    from vox.config import IMAGE_NSFW_FILTER
    if IMAGE_NSFW_FILTER.lower() == "off":
        pipe_kwargs["safety_checker"] = None
        pipe_kwargs["requires_safety_checker"] = False
    if "stabilityai" in model_id.lower():
        pipe_kwargs["variant"] = "fp16"
        pipe_kwargs["use_safetensors"] = True
    log.info("Loading diffusers-format model: %s", model_id)
    try:
        return pipeline_cls.from_pretrained(model_id, **pipe_kwargs)
    except (EnvironmentError, OSError) as e:
        if "safetensors" in str(e).lower():
            log.warning("Safetensors load failed, retrying with use_safetensors=False: %s", e)
            pipe_kwargs["use_safetensors"] = False
            return pipeline_cls.from_pretrained(model_id, **pipe_kwargs)
        raise


def _extract_image_count(text: str) -> int:
    """Extract requested image count from text, capped at 10."""
    m = re.search(r"\b(\d+)\s+(image|picture|photo|pic|pics|selfie)", text, re.IGNORECASE)
    if m:
        return min(int(m.group(1)), 10)
    return 1


@_register("generate_image")
def _generate_image(prompt: str = "", style: str = "", _selfie: bool = False,
                    _nsfw_unlocked: bool = False, count: int = 1, **kwargs) -> str:
    """Generate an image using dual-model routing: SDXL for SFW, Juggernaut for NSFW/persona."""
    from vox.config import (
        DOWNLOADS_DIR,
        IMAGE_CFG,
        IMAGE_HEIGHT,
        IMAGE_MODEL,
        IMAGE_MODEL_NSFW,
        IMAGE_NEGATIVE_PROMPT,
        IMAGE_NSFW_FILTER,
        IMAGE_STEPS,
        IMAGE_WIDTH,
    )

    if not prompt:
        return "No image prompt provided."

    # Rate limit — prevent rapid-fire requests from overwhelming GPU
    global _last_image_gen_time
    now = _time.monotonic()
    elapsed = now - _last_image_gen_time
    if _last_image_gen_time > 0 and elapsed < _IMAGE_COOLDOWN_SEC:
        wait = _IMAGE_COOLDOWN_SEC - elapsed
        log.info("Image gen cooldown: waiting %.1fs", wait)
        _time.sleep(wait)
    _last_image_gen_time = _time.monotonic()

    # For selfie requests, the prompt is already built by _build_persona_prompt.
    # Reinforce gender to prevent SDXL from generating the wrong sex.
    if _selfie:
        full_prompt = f"1girl, solo, female, {prompt}"
    elif style:
        full_prompt = f"{prompt}, {style}"
    else:
        full_prompt = prompt

    # Dual-model routing: NSFW/persona → Juggernaut, SFW → SDXL base
    use_nsfw = _should_use_nsfw_model(full_prompt, _selfie, _nsfw_unlocked)
    model_id = IMAGE_MODEL_NSFW if use_nsfw else IMAGE_MODEL
    log.info("generate_image: prompt=%r, selfie=%s, nsfw_route=%s, model=%s",
             prompt, _selfie, use_nsfw, model_id)
    log.info("generate_image: steps=%d, cfg=%.1f, size=%dx%d, nsfw_filter=%s",
             IMAGE_STEPS, IMAGE_CFG, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_NSFW_FILTER)

    try:
        import torch
    except ImportError:
        log.error("generate_image: torch not installed")
        return "Image generation requires PyTorch. Install with: pip install vox[image]"

    # VRAM check — SDXL needs ~12GB, warn early rather than OOM mid-generation
    if torch.cuda.is_available():
        free_vram, total_vram = torch.cuda.mem_get_info()
        free_mb = free_vram / (1024 * 1024)
        min_required = 8000  # MB — conservative minimum for SDXL with attention slicing
        if free_mb < min_required:
            log.warning("Low VRAM for image gen: %.0f MB free (need ~%d MB)", free_mb, min_required)
            # Try to free cached models first
            torch.cuda.empty_cache()
            free_vram, _ = torch.cuda.mem_get_info()
            free_mb = free_vram / (1024 * 1024)
            if free_mb < min_required:
                return (f"Not enough GPU memory for image generation "
                        f"({free_mb:.0f} MB free, need ~{min_required} MB). "
                        f"Try closing other GPU tasks first.")

    # Prompt length guard — SDXL tokenizer max is 77 tokens per encoder
    words = full_prompt.split()
    if len(words) > 120:
        full_prompt = " ".join(words[:120])
        log.warning("Prompt truncated from %d to 120 words", len(words))

    # Both Juggernaut and SDXL base use StableDiffusionXLPipeline
    is_sdxl = "xl" in model_id.lower() or "sdxl" in model_id.lower() or "juggernaut" in model_id.lower()

    # Determine LoRA path for selfie requests
    lora_dir = None
    if _selfie:
        from vox.lora import get_lora_path
        from vox.persona import get_card
        card = get_card()
        if card:
            lora_dir = get_lora_path(card["name"])

    # Model manager: load on demand, unload after generation to free VRAM.
    # Cache key encodes model+lora combo so different configs get separate entries.
    cache_key = f"sdxl:{model_id}:{lora_dir or ''}"

    def _build_pipeline():
        """Loader function for model_manager — creates and configures the pipeline."""
        from pathlib import Path
        if is_sdxl:
            from diffusers import StableDiffusionXLPipeline
            log.info("Loading SDXL pipeline: %s", model_id)
            p = _load_pipeline(StableDiffusionXLPipeline, model_id, torch.float16)
        else:
            from diffusers import StableDiffusionPipeline
            log.info("Loading SD 1.5 pipeline: %s", model_id)
            pipe_kwargs = {"torch_dtype": torch.float16}
            if IMAGE_NSFW_FILTER.lower() == "off":
                pipe_kwargs["safety_checker"] = None
            p = StableDiffusionPipeline.from_pretrained(model_id, **pipe_kwargs)

        p = p.to("cuda")
        p.enable_attention_slicing()

        # Auto-load LoRA if trained for this persona
        if lora_dir:
            try:
                lora_unet = Path(lora_dir) / "unet"
                if lora_unet.exists():
                    from peft import PeftModel
                    p.unet = PeftModel.from_pretrained(
                        p.unet, str(lora_unet), adapter_name="persona",
                    )
                    p.unet.set_adapter("persona")
                    # Scale adapter — 0.85 balances likeness vs prompt adherence
                    try:
                        p.unet.set_adapter_scale({"persona": 0.85})
                    except (AttributeError, TypeError):
                        pass  # older peft versions may not support this
                    log.info("LoRA loaded via PEFT (scale=0.85) from %s", lora_unet)
                else:
                    p.load_lora_weights(str(lora_dir))
                    p.fuse_lora(lora_scale=0.85)
                    log.info("LoRA loaded via diffusers (scale=0.85) from %s", lora_dir)
            except Exception as e:
                log.warning("Failed to load LoRA: %s", e)
        return p

    # Register if not already known, then acquire
    if cache_key not in _mm.status():
        _mm.register(cache_key, _build_pipeline, vram_mb=12000, keep_alive=120)

    try:
        pipe = _mm.acquire(cache_key)

        # Split long prompts for SDXL dual encoder — subject in prompt,
        # style/quality in prompt_2 (each gets 77 tokens)
        prompt_1 = full_prompt
        prompt_2 = None
        if is_sdxl and len(full_prompt.split()) > 40:
            # Find a natural split point — style tags usually start after the scene
            style_markers = ["photorealistic", "canon", "shot on", "natural lighting", "shallow depth", "film grain", "raw photo", "8k"]
            split_idx = len(full_prompt)
            for marker in style_markers:
                idx = full_prompt.lower().find(marker)
                if idx > 0 and idx < split_idx:
                    split_idx = idx
            if split_idx < len(full_prompt):
                prompt_1 = full_prompt[:split_idx].rstrip(", ")
                prompt_2 = full_prompt[split_idx:]
                log.info("Split prompt for SDXL dual encoder: p1=%d chars, p2=%d chars", len(prompt_1), len(prompt_2))

        # Selfie negative prompt: add male-exclusion terms to prevent wrong gender
        neg_prompt = IMAGE_NEGATIVE_PROMPT or ""
        if _selfie:
            male_neg = "male, man, boy, masculine, male body, male anatomy"
            neg_prompt = f"{male_neg}, {neg_prompt}" if neg_prompt else male_neg
            # SFW selfies: actively push model away from revealing content
            if not _nsfw_unlocked:
                sfw_neg = "nude, naked, topless, nsfw, cleavage, sports bra, underwear, lingerie, suggestive"
                neg_prompt = f"{sfw_neg}, {neg_prompt}"
        log.info("Generating image: %r (negative: %r)", prompt_1[:120], neg_prompt[:80])
        gen_kwargs = {
            "negative_prompt": neg_prompt or None,
        }
        if prompt_2 and is_sdxl:
            gen_kwargs["prompt_2"] = prompt_2

        # Progress callback for step-by-step feedback
        actual_count = max(1, min(count, 10))
        total_steps = IMAGE_STEPS * actual_count
        steps_done = 0
        if _image_progress_fn:
            def _step_callback(pipe_self, step, timestep, callback_kwargs):
                try:
                    _image_progress_fn(steps_done + step + 1, total_steps)
                except Exception:
                    pass
                return callback_kwargs
            gen_kwargs["callback_on_step_end"] = _step_callback

        filenames = []
        for img_idx in range(actual_count):
            # Vary seed per image for diversity
            generator = torch.Generator(device="cuda").manual_seed(
                int(_time.time() * 1000) + img_idx
            )
            try:
                result = pipe(
                    prompt_1,
                    **gen_kwargs,
                    guidance_scale=IMAGE_CFG,
                    num_inference_steps=IMAGE_STEPS,
                    width=IMAGE_WIDTH,
                    height=IMAGE_HEIGHT,
                    generator=generator,
                )
            except torch.cuda.OutOfMemoryError:
                log.warning("CUDA OOM during generation — clearing cache and retrying at lower res")
                torch.cuda.empty_cache()
                retry_w = (IMAGE_WIDTH * 3 // 4) // 8 * 8
                retry_h = (IMAGE_HEIGHT * 3 // 4) // 8 * 8
                result = pipe(
                    prompt_1,
                    **gen_kwargs,
                    guidance_scale=IMAGE_CFG,
                    num_inference_steps=IMAGE_STEPS,
                    width=retry_w,
                    height=retry_h,
                    generator=generator,
                )
            image = result.images[0]
            steps_done += IMAGE_STEPS

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = f"_{img_idx}" if actual_count > 1 else ""
            filename = f"vox_image_{timestamp}{suffix}.png"
            filepath = DOWNLOADS_DIR / filename
            image.save(filepath)
            filenames.append(filename)
            log.info("Image %d/%d saved to %s", img_idx + 1, actual_count, filepath)
            # Push image to WebSocket immediately — don't rely on post-hoc scan
            if _image_saved_fn:
                try:
                    _image_saved_fn(filename)
                except Exception:
                    pass

        # Release pipeline — frees ~12GB VRAM back to chat mode.
        # keep_alive=120s means rapid re-requests reuse the cached pipeline.
        _mm.release(cache_key)

        if len(filenames) == 1:
            return f"Image generated and saved to {filenames[0]}"
        saved_list = ", ".join(filenames)
        return f"{len(filenames)} images generated and saved to {saved_list}"

    except ImportError:
        log.error("generate_image: diffusers not installed")
        return "Image generation requires the 'diffusers' package. Install with: pip install vox[image]"
    except Exception as e:
        log.exception("generate_image failed: %s", e)
        _mm.release(cache_key, force=True)  # Ensure cleanup on error
        return f"Image generation failed: {e}"


# ---------------------------------------------------------------------------
# Response caching — TTL-based cache for tool results (#36)
# ---------------------------------------------------------------------------

_TOOL_TTL: dict[str, int] = {
    "get_weather": 1800,       # 30 min
    "web_search": 3600,        # 1 hour
    "get_system_info": 300,    # 5 min
    "get_current_time": 30,    # 30 sec
}

_tool_cache: dict[str, tuple[float, str]] = {}


def _cache_key(name: str, args: dict) -> str:
    """Build a cache key from tool name and args (excluding internal args)."""
    clean = {k: v for k, v in sorted(args.items()) if not k.startswith("_")}
    return f"{name}:{clean}"


def _cache_get(name: str, args: dict) -> str | None:
    """Get cached result if TTL hasn't expired."""
    ttl = _TOOL_TTL.get(name)
    if ttl is None:
        return None
    key = _cache_key(name, args)
    entry = _tool_cache.get(key)
    if entry is None:
        return None
    timestamp, result = entry
    if _time.time() - timestamp > ttl:
        del _tool_cache[key]
        return None
    log.info("Cache hit for %s (age %.0fs)", name, _time.time() - timestamp)
    return result


def _cache_set(name: str, args: dict, result: str):
    """Store result in cache if tool has a TTL."""
    if name in _TOOL_TTL:
        _tool_cache[_cache_key(name, args)] = (_time.time(), result)


def cache_bust(name: str | None = None):
    """Clear cache for a specific tool or all tools."""
    if name:
        keys = [k for k in _tool_cache if k.startswith(f"{name}:")]
        for k in keys:
            del _tool_cache[k]
    else:
        _tool_cache.clear()


def execute_tool(name: str, args: dict) -> str:
    """Execute a registered tool by name, with TTL caching."""
    log.info("execute_tool: %s(%s)", name, args)
    fn = _TOOL_REGISTRY.get(name)
    if fn is None:
        log.warning("Unknown tool: %s", name)
        return f"Unknown tool: {name}"

    # Check cache (skip for tools that mutate state)
    cached = _cache_get(name, args)
    if cached is not None:
        return cached

    try:
        result = fn(**args)
        log.info("execute_tool %s result (%d chars): %s", name, len(result), result[:200])
        _cache_set(name, args, result)
        return result
    except Exception as e:
        log.exception("Tool error in %s: %s", name, e)
        return f"Tool error: {e}"


# ---------------------------------------------------------------------------
# Plugin system — load plugins at import time
# ---------------------------------------------------------------------------

def _init_plugins():
    """Load and register plugins from plugin directories."""
    try:
        import sys

        from vox import plugins
        count = plugins.register_with_tools(sys.modules[__name__])
        if count:
            log.info("Registered %d plugin tool(s)", count)
    except Exception as e:
        log.debug("Plugin init: %s", e)


_init_plugins()
