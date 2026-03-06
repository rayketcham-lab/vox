# VOX Feature Roadmap Research

> Competitive analysis of personal AI assistants (open-source and commercial) as of March 2026.
> Features VOX already has are noted with [HAVE]. Gaps and opportunities are the focus.

---

## MUST HAVE (we're embarrassing without these)

### 1. Smarter Conversation Memory (Vector DB + Episodic)
- **What**: Replace flat JSON memory with a tiered system: core memory (always in context), recall memory (searchable conversation history), archival memory (vector-indexed long-term knowledge). This is what MemGPT/Letta pioneered.
- **Why**: Every serious competitor (Replika, ChatGPT, Gemini) remembers context across sessions. Flat JSON doesn't scale, can't do semantic search ("what did I say about that restaurant last month?"), and wastes context window.
- **Complexity**: Medium
- **Libs**: ChromaDB (easiest local vector DB, SQLite-like persistence), FAISS (Meta, fastest for pure search), sentence-transformers for local embeddings, LlamaIndex for RAG pipeline. All run locally, no cloud needed.
- **VOX has**: Basic "remember X" key-value memory. Needs upgrade to semantic search + tiered architecture.

### 2. Calendar / Scheduling Integration
- **What**: Read/write Google Calendar (or CalDAV for local-first). "What's my day look like?" / "Schedule a dentist appointment Tuesday at 3" / "Move my 2pm to 4pm" / conflict detection / smart scheduling suggestions.
- **Why**: Apple Intelligence, Google Gemini, and every commercial assistant does this. A personal assistant that can't manage your schedule is a toy. Rabbit R1's community skills include calendar triage.
- **Complexity**: Medium (Google Calendar API is well-documented; OAuth is the annoying part)
- **Libs**: `google-api-python-client` + `google-auth-oauthlib` for Google Calendar; `caldav` library for self-hosted (Nextcloud/Radicale); `python-dateutil` + `dateparser` for natural language dates ("next Tuesday", "in two weeks").
- **VOX has**: Reminders/timers only. No calendar awareness.

### 3. Music / Media Control
- **What**: "Play jazz", "skip this song", "turn it up", "play my Discover Weekly", "what's playing?" Control Spotify, local media, or smart speakers.
- **Why**: One of the most-used voice assistant features globally. Alexa/Google/Siri all do this. It's table-stakes for daily use.
- **Complexity**: Easy-Medium
- **Libs**: `spotipy` (Spotify Web API wrapper), MCP Spotify server (for tool-calling pattern), `python-vlc` for local media, Home Assistant media_player entities for smart speakers. Music Assistant is an open-source aggregator.
- **VOX has**: Nothing for media control.

### 4. Smart Home Integration (Home Assistant / MQTT)
- **What**: "Turn off the living room lights", "set thermostat to 72", "lock the front door", "is the garage open?" Control any Home Assistant entity via its REST API or WebSocket API.
- **Why**: This is the #1 reason people buy voice assistants. Home Assistant has 2000+ integrations. Matter protocol means any modern smart device works. Google is replacing Assistant with Gemini for home control in 2026.
- **Complexity**: Easy (Home Assistant REST API is trivial; MQTT is pub/sub)
- **Libs**: `homeassistant-api` or just raw REST calls to HA's API, `paho-mqtt` for MQTT, `python-matter-server` for direct Matter (but HA abstracts this). A single tool function calling HA's `/api/services/<domain>/<service>` covers 90% of use cases.
- **VOX has**: Nothing for smart home.

### 5. Interruption Handling / Barge-In
- **What**: User can interrupt VOX mid-sentence and it stops talking immediately, processes the new input. No waiting for TTS to finish.
- **Why**: ChatGPT Advanced Voice Mode, Gemini Live, and every phone call does this. Without it, conversations feel robotic and frustrating. This is the #1 UX complaint about voice assistants.
- **Complexity**: Medium (requires streaming TTS with cancel, concurrent audio monitoring)
- **Libs**: Already have sounddevice. Need TTS streaming (chunk-by-chunk playback with abort capability), VAD running during playback (Silero VAD or webrtcvad).
- **VOX has**: Sequential pipeline (STT -> LLM -> TTS). No barge-in.

### 6. Streaming Responses (LLM -> TTS)
- **What**: Start speaking the first sentence while the LLM is still generating the rest. Don't wait for full response before TTS begins.
- **Why**: Reduces perceived latency by 2-5x. ChatGPT voice mode, Gemini Live, and every competitive assistant does this. Without it, there's an awkward 3-8 second silence.
- **Complexity**: Medium (sentence boundary detection, TTS queue, async pipeline)
- **Libs**: Ollama streaming API (already supported), sentence tokenization (nltk or regex), async TTS queue.
- **VOX has**: Waits for full LLM response before TTS. Pipeline is sequential.

---

## SHOULD HAVE (real differentiators)

### 7. Autonomous Multi-Step Task Execution
- **What**: "Book me a flight to Denver for next weekend under $300" -> searches flights, compares options, presents top 3, books on approval. "Research the best standing desks under $500 and make a comparison table." Multi-step planning with tool chaining.
- **Why**: AutoGPT proved the concept. OpenAI Operator, Google Chrome auto-browse, and Perplexity Comet all do this in 2026. Rabbit R1's LAM (Large Action Model) was built around this idea. The agent decomposes goals into subtasks, executes them, evaluates results, and adjusts.
- **Complexity**: Hard
- **Libs**: `browser-use` (60k+ GitHub stars, Playwright-based browser automation for AI agents), LangChain/LangGraph for agent orchestration, or roll your own with Ollama tool-calling chains. OpenAI Agents SDK pattern is instructive.
- **VOX has**: Single-step tool calling. No multi-step planning or chaining.

### 8. Contact / Relationship Management
- **What**: "When's Mom's birthday?", "Remind me to call Dave every 2 weeks", "Who did I last email about the project?", "Add Sarah - met her at the conference, works at Acme." Track contacts, relationships, interaction history, birthdays, and follow-up cadences.
- **Why**: Personal CRMs (Monica, Clay, folk) are a hot category. Replika tracks relationship context. A personal assistant that knows your social graph is dramatically more useful.
- **Complexity**: Medium (SQLite + some CRUD tools)
- **Libs**: SQLite or the existing memory system extended with contact schema. `vobject` for vCard import/export. Monica CRM is open-source (PHP) for inspiration. Could integrate with Google Contacts API.
- **VOX has**: Basic user memory. No structured contact/relationship tracking.

### 9. Screen / Visual Awareness
- **What**: "What's on my screen?", "Summarize this article I'm reading", "Fill in this form", "What error is showing?" Take a screenshot, OCR/analyze it, and act on it.
- **Why**: Apple Intelligence's "on-screen awareness" is their flagship 2026 Siri feature. Open Interpreter's Computer API does this. Qwen3-VL can operate GUIs. This is where assistants go from "voice search" to "digital coworker."
- **Complexity**: Medium-Hard
- **Libs**: `mss` or `Pillow` for screenshots, local VLMs (Qwen3-VL, LLaMA 3.2 Vision, GLM-4.6V via Ollama — llava or minicpm-v), `pytesseract` for basic OCR. Could use Claude API vision for complex analysis.
- **VOX has**: Image generation, but no screen reading or visual input.

### 10. Real-Time Voice Translation
- **What**: "Translate everything I say into Spanish" -> continuous live translation. Or two-way: "Help me talk to this French speaker."
- **Why**: Gemini Live now does speech-to-speech translation natively. It's a killer feature for travel and multilingual households. Whisper already supports 99 languages.
- **Complexity**: Medium
- **Libs**: Faster-Whisper (already have, supports 99 languages), translation models (Helsinki-NLP/OPUS-MT via transformers, or `argostranslate` for fully local), XTTS for target-language synthesis.
- **VOX has**: STT in multiple languages via Whisper, but no translation pipeline.

### 11. Better TTS Quality (Kokoro / Sesame CSM)
- **What**: Upgrade from Piper (fast but robotic) and XTTS (good but slow/heavy) to newer models that sound more natural. Kokoro-82M does 210x realtime on GPU with near-human quality. Sesame CSM uses conversation history for more natural prosody.
- **Why**: Voice quality is the #1 factor in whether people actually enjoy talking to an assistant. ChatGPT and Gemini Live have set the bar very high for natural intonation, pauses, and expressiveness.
- **Complexity**: Easy-Medium
- **Libs**: Kokoro-82M (Apache 2.0, 54 voices, 8 languages, 82M params, runs on CPU even), Sesame CSM-1B (Apache 2.0, conversation-aware, needs GPU). Both available on HuggingFace.
- **VOX has**: Piper (fast/robotic) and XTTS v2 (voice-clone quality but heavy). Could add Kokoro as a third option.

### 12. Emotion / Sentiment Detection from Voice
- **What**: Detect user's emotional state from voice (stressed, happy, tired, frustrated) and adapt responses accordingly. "You sound tired, want me to keep it brief?"
- **Why**: Replika and Character.AI adapt to emotional context. It makes the persona feel genuinely responsive rather than performing a script. Combined with the persona mood system VOX already has, this would be powerful.
- **Complexity**: Medium
- **Libs**: `speechbrain` (SER models), `transformers` (Wav2Vec2 fine-tuned for emotion), Silero models. Can extract pitch/energy/tempo from audio features with `librosa`. RAVDESS dataset for training.
- **VOX has**: Persona moods and activities, but they're script-driven, not reactive to user's actual emotional state.

### 13. MCP (Model Context Protocol) Support
- **What**: Implement MCP server/client so VOX can use the growing ecosystem of 5,800+ MCP servers (Slack, GitHub, Google Drive, Postgres, etc.) as tools, and expose VOX's capabilities as an MCP server for other agents.
- **Why**: MCP is now backed by Anthropic, OpenAI, Google, and Microsoft. 97M+ monthly SDK downloads. It's becoming THE standard for AI tool integration. Instead of building every integration from scratch, just connect MCP servers.
- **Complexity**: Medium
- **Libs**: `mcp` Python SDK (official), existing MCP servers on npm/PyPI for Spotify, Google Calendar, Home Assistant, Slack, etc. One integration pattern to rule them all.
- **VOX has**: Custom tool-calling via Ollama. No MCP support.

---

## NICE TO HAVE (wow factor)

### 14. Location Awareness / Geofencing
- **What**: Know where the user is. "Remind me to buy milk when I'm near Trader Joe's." Auto-switch modes (work vs. home). Proactive: "Traffic is heavy on your commute, leave 15 minutes early."
- **Why**: Google/Apple assistants use location constantly. Makes proactive behaviors much smarter. Geofencing triggers can automate routines.
- **Complexity**: Medium (need phone companion app or GPS device)
- **Libs**: `geopy` for geocoding, `shapely` for geofence geometry, Google Maps API or `openrouteservice` for routing/traffic. Phone app would use native location APIs. Home Assistant can provide device_tracker entities.
- **VOX has**: Nothing for location. Would need mobile companion or HA device tracker.

### 15. Health / Fitness Tracking Integration
- **What**: "How did I sleep last night?", "What's my step count?", "Am I hitting my calorie goal?" Pull data from Apple Health, Google Health Connect, Fitbit, Garmin, etc.
- **Why**: 450M+ smartwatch users worldwide. Health awareness makes an assistant feel holistic. Morning briefings could include sleep quality and activity reminders.
- **Complexity**: Medium-Hard (Google Fit deprecated July 2025, must use Health Connect; Apple Health requires MCP server or export)
- **Libs**: Terra API (unified health data API for all wearables, has Python SDK), Apple Health MCP Server (open source), Health Connect via Android REST. Fitbit has a well-documented Web API.
- **VOX has**: Nothing for health data.

### 16. Financial Awareness
- **What**: "How much did I spend on food this month?", "Am I on budget?", "What's my account balance?" Read-only financial data with AI-powered spending analysis.
- **Why**: Every premium assistant is adding financial features. It's deeply personal and high-value. Even basic "you've spent $X this week" is useful.
- **Complexity**: Hard (Plaid API costs money; security requirements are serious; PII handling)
- **Libs**: Plaid API (bank aggregation, $$ for production), `ofxparse` for OFX file import (free), CSV import from bank exports. SQLite for transaction storage. Keep it read-only and local for security.
- **VOX has**: Nothing for finance.

### 17. Document / File Intelligence
- **What**: "Summarize that PDF I downloaded yesterday", "Find the contract with the warranty terms", "What did the email from Dave say about the deadline?" Index and search local files with RAG.
- **Why**: Apple Intelligence's personal context feature is exactly this. Being able to search your own documents by meaning (not just filename) is transformative.
- **Complexity**: Medium
- **Libs**: `unstructured` (parses PDF/DOCX/HTML/email), ChromaDB for vector indexing, `watchdog` for filesystem monitoring, `pytesseract` for OCR on images. LlamaIndex has a file-system reader.
- **VOX has**: Nothing for local file awareness.

### 18. Meeting / Conversation Transcription
- **What**: Record meetings, calls, or in-person conversations. Transcribe, summarize, extract action items. Like Limitless Pendant but software-only.
- **Why**: Limitless was acquired by Meta for this exact feature. ChatGPT Record does this. It's one of the highest-value AI features for professionals.
- **Complexity**: Easy-Medium (already have Whisper STT)
- **Libs**: Faster-Whisper (already have), `pyannote-audio` for speaker diarization (who said what), system audio capture via `sounddevice` loopback or virtual audio cable.
- **VOX has**: STT for commands. Not set up for long-form transcription or diarization.

### 19. Learning / Adaptation Over Time
- **What**: Track what the user asks for, when, and how. Learn patterns: "User always asks for weather at 7am" -> proactively provide it. "User prefers concise answers" -> auto-adjust verbosity. Implicit preference learning.
- **Why**: This is what separates a great assistant from a good one. Replika does this. Apple Intelligence promises "personal context." The assistant should get better the more you use it.
- **Complexity**: Hard (needs interaction logging, pattern detection, preference modeling)
- **Libs**: SQLite for interaction logs, scikit-learn for basic pattern detection, or simply prompt-engineer the LLM with aggregated usage stats. MemGPT's self-editing memory is a simpler version of this.
- **VOX has**: Manual memory ("remember X"). No automatic pattern learning.

### 20. Multi-User / Voice ID
- **What**: Recognize different household members by voice. Switch persona context, memory, and preferences per user. "Good morning Ray" vs. "Good morning Ann."
- **Why**: Every smart speaker does this (Alexa Voice Profiles, Google Voice Match). Without it, a household assistant is limited to one person.
- **Complexity**: Medium
- **Libs**: `pyannote-audio` for speaker embedding/verification, `resemblyzer` for voice fingerprinting. Enroll speakers with a few seconds of speech, then classify in real-time.
- **VOX has**: Single-user. No voice identification.

---

## BLEEDING EDGE (flex territory)

### 21. Browser Automation Agent
- **What**: Full web browsing agent that can navigate sites, fill forms, click buttons, extract data. "Go to Amazon and find the cheapest USB-C hub with at least 4 ports." "Fill out this job application for me."
- **Why**: OpenAI Operator, Google Chrome auto-browse, Perplexity Comet, and browser-use are all competing here. This is the frontier of AI agents in 2026. The market is projected to grow from $4.5B to $76.8B by 2034.
- **Complexity**: Hard
- **Libs**: `browser-use` (open source, 60k+ stars, Playwright + LLM integration), `playwright` directly, Browserbase for hosted browser environments. Works with local LLMs via LangChain.
- **VOX has**: Web fetch (raw HTML grab). No interactive browsing.

### 22. Always-On Ambient Listening (Limitless-style)
- **What**: Continuously record ambient audio (with consent), transcribe in background, build searchable life log. "What did my doctor say about the dosage?" "What was the name of that restaurant someone mentioned?"
- **Why**: Meta paid to acquire Limitless for this. It's the ultimate memory augmentation. Combined with vector search, it becomes a "rewind" for your life.
- **Complexity**: Hard (privacy, storage, continuous STT, speaker diarization)
- **Libs**: Whisper (have it), `pyannote-audio` for diarization, ChromaDB for indexing, aggressive compression (Opus codec for storage). Need clear opt-in consent UX.
- **VOX has**: Wake-word triggered listening only.

### 23. Vision + Camera Integration
- **What**: "What's this?" (point phone camera). "Read this label." "Identify this plant." "What's wrong with my car's dashboard light?" Real-time camera feed analysis.
- **Why**: Rabbit R1 has a rotating camera for this. Humane Pin had it. Gemini Live does it on phone. Multimodal is the future.
- **Complexity**: Hard (needs camera feed + real-time VLM inference)
- **Libs**: OpenCV for camera capture, Qwen3-VL or LLaMA 3.2 Vision via Ollama, `llava` model family. Could work via the web UI with phone camera (MediaDevices API).
- **VOX has**: Image generation. No camera/vision input.

### 24. Voice Cloning On-Demand
- **What**: Clone any voice from a short sample. "Read this email in Morgan Freeman's voice." Or more practically: "Read this in Mom's voice" for accessibility/comfort.
- **Why**: XTTS already supports voice cloning. Sesame CSM can generate with custom voices. This is a natural extension of the persona system.
- **Complexity**: Medium (already have XTTS)
- **Libs**: XTTS v2 (already have, supports voice cloning from ~6 seconds of audio), Sesame CSM, OpenVoice v2 (cross-lingual voice cloning), RVC for voice conversion.
- **VOX has**: XTTS voice cloning for persona. Could generalize to arbitrary voices.

### 25. Proactive Intelligence / Predictive Actions
- **What**: The assistant anticipates needs before being asked. "Your meeting starts in 10 minutes and traffic is heavy — want me to send a 'running late' message?" "You usually order groceries on Fridays — want me to start your list?" "Your electricity bill is 30% higher than last month."
- **Why**: This is what separates an assistant from a search engine. Google Now pioneered this concept. Apple Intelligence promises it. It requires calendar + location + habits + pattern learning all working together.
- **Complexity**: Hard (requires multiple data sources and pattern learning from #19)
- **Libs**: Combination of calendar, location, memory, and a rule engine or learned patterns. Could start simple with time-based rules and expand.
- **VOX has**: Morning briefings, check-ins, goodnight. These are scripted, not truly predictive.

### 26. End-to-End Voice Model (No Pipeline)
- **What**: Replace the STT -> LLM -> TTS pipeline with a single model that takes audio in and produces audio out. Dramatically lower latency, more natural conversation flow, better handling of tone/emotion/interruption.
- **Why**: GPT-4o's native audio mode and Gemini 2.5 Flash Native Audio do this. Sesame CSM is a step toward open-source versions. This is where voice AI is heading.
- **Complexity**: Very Hard (models are bleeding-edge, VRAM hungry, not production-ready in open source)
- **Libs**: Sesame CSM-1B (closest open-source option), Moshi by Kyutai (open-source speech-to-speech), Gazelle by Tincans AI. None match GPT-4o quality yet.
- **VOX has**: Traditional pipeline. This is a long-term bet.

---

## Priority Implementation Order (Recommended)

### Phase 1: Core UX (make daily use not painful)
1. **Streaming LLM -> TTS** (#6) — biggest perceived latency win, medium effort
2. **Barge-in / Interruption** (#5) — required for natural conversation
3. **Kokoro TTS option** (#11) — easy win, dramatic quality improvement
4. **Smart Home / Home Assistant** (#4) — easy integration, daily use

### Phase 2: Actually Useful (from toy to tool)
5. **Calendar integration** (#2) — table stakes for personal assistant
6. **Music/media control** (#3) — most-requested voice assistant feature
7. **Vector memory upgrade** (#1) — enables everything else to get smarter
8. **MCP support** (#13) — unlocks 5,800+ tool integrations

### Phase 3: Differentiation (why VOX, not Alexa)
9. **Contact/relationship management** (#8)
10. **Multi-step task execution** (#7)
11. **Screen awareness** (#9)
12. **Emotion detection** (#12)

### Phase 4: Advanced
13. **Document intelligence** (#17)
14. **Meeting transcription** (#18)
15. **Location awareness** (#14)
16. **Learning/adaptation** (#19)
17. **Multi-user voice ID** (#20)
18. **Real-time translation** (#10)

### Phase 5: Frontier
19. **Browser automation** (#21)
20. **Health/fitness** (#15)
21. **Financial awareness** (#16)
22. **Vision/camera** (#23)
23. **Ambient listening** (#22)
24. **Predictive intelligence** (#25)
25. **End-to-end voice model** (#26)

---

## Sources

- [Humane AI Pin dead, Rabbit R1 survival](https://www.techradar.com/computing/artificial-intelligence/with-the-humane-ai-pin-now-dead-what-does-the-rabbit-r1-need-to-do-to-survive)
- [Apple Intelligence Siri 2026 upgrade](https://ia.acs.org.au/article/2026/apple-reveals-the-ai-behind-siri-s-big-2026-upgrade.html)
- [Siri personal context delayed to Spring 2026](https://www.macrumors.com/2025/06/12/apple-intelligence-siri-spring-2026/)
- [Gemini replacing Google Assistant in 2026](https://9to5google.com/2025/12/19/google-assistant-gemini-2026/)
- [Gemini Live audio model upgrades](https://blog.google/products/gemini/gemini-audio-model-updates/)
- [ChatGPT voice mode unified interface](https://techcrunch.com/2025/11/25/chatgpts-voice-mode-is-no-longer-a-separate-interface/)
- [ChatGPT 2026 features guide](https://ai-basics.com/chatgpt-update-new-features-guide/)
- [MemGPT/Letta memory architecture](https://docs.letta.com/concepts/memgpt/)
- [Letta agent memory deep dive](https://medium.com/@piyush.jhamb4u/stateful-ai-agents-a-deep-dive-into-letta-memgpt-memory-models-a2ffc01a7ea1)
- [Top 10 AI memory products 2026](https://medium.com/@bumurzaqov2/top-10-ai-memory-products-2026-09d7900b5ab1)
- [Home Assistant MQTT](https://www.home-assistant.io/integrations/mqtt/)
- [Home Assistant Matter](https://www.home-assistant.io/integrations/matter/)
- [MCP Spotify Player](https://github.com/vsaez/mcp-spotify-player)
- [Music Assistant](https://www.music-assistant.io/blog/2025/03/05/music-assistants-next-big-hit/)
- [Open Interpreter](https://github.com/openinterpreter/open-interpreter)
- [browser-use](https://github.com/browser-use/browser-use)
- [2025-2026 AI computer-use agents guide](https://o-mega.ai/articles/the-2025-2026-guide-to-ai-computer-use-benchmarks-and-top-ai-agents)
- [Agentic AI browsers 2026](https://www.kdnuggets.com/the-best-agentic-ai-browsers-to-look-for-in-2026)
- [AutoGPT deep dive](https://axis-intelligence.com/autogpt-deep-dive-use-cases-best-practices/)
- [Character.AI and Replika comparison](https://www.cyberlink.com/blog/trending-topics/3932/ai-companion-app)
- [Replika review 2026](https://charalt.com/replika-review/)
- [Jan.ai open-source ChatGPT replacement](https://www.jan.ai/)
- [Personal CRM tools 2025](https://wavecnct.com/blogs/news/the-6-best-personal-crm-tools-in-2025)
- [Monica open-source personal CRM](https://www.folk.app/articles/best-ai-personal-crm)
- [Kokoro-82M TTS](https://huggingface.co/hexgrad/Kokoro-82M)
- [Sesame CSM](https://github.com/SesameAILabs/csm)
- [Best open-source TTS 2026](https://www.bentoml.com/blog/exploring-the-world-of-open-source-text-to-speech-models)
- [Limitless AI pendant](https://www.limitless.ai/)
- [Meta acquires Limitless](https://www.marketplace.org/story/2025/10/23/whats-it-like-to-use-wearable-ai-tech)
- [Multimodal VLMs 2026](https://www.bentoml.com/blog/multimodal-ai-a-guide-to-open-source-vision-language-models)
- [ChromaDB vs FAISS for RAG](https://medium.com/@priyaskulkarni/vector-databases-for-rag-faiss-vs-chroma-vs-pinecone-6797bd98277d)
- [Whisper v4 + Llama 4 local assistant](https://markaicode.com/build-local-voice-assistant-whisper-llama4/)
- [MCP specification](https://modelcontextprotocol.io/specification/2025-11-25)
- [MCP enterprise adoption](https://guptadeepak.com/the-complete-guide-to-model-context-protocol-mcp-enterprise-adoption-market-trends-and-implementation-strategies/)
- [Google Calendar AI assistant with ADK](https://medium.com/google-cloud/build-your-own-ai-google-calendar-assistant-with-agent-development-kit-29f917be9e07)
- [Terra health data API](https://tryterra.co)
- [Google Fit deprecated](https://developers.google.com/fit)
- [Plaid personal finance API](https://plaid.com/use-cases/personal-finances/)
- [AI personal finance agent](https://olikhatib.substack.com/p/build-an-ai-personal-finance-agent)
- [Geofencing with Python](https://johal.in/location-based-services-geofencing-with-python-and-google-maps-api/)
- [PersonalLLM personalization research (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/a730abbcd6cf4a371ca9545db5922442-Paper-Conference.pdf)
- [Self-improving data agents](https://powerdrill.ai/blog/self-improving-data-agents)
