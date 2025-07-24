# Twinly API

Eine saubere FastAPI-Anwendung für LLM-basierte Chatbots, Workflows und Agents.

## 🏗️ Projekt-Struktur

```
/app
├── __init__.py
├── main.py                 ← FastAPI App Initialisierung
├── /api                    ← FastAPI Routen
│   ├── __init__.py
│   └── chat.py             ← Chat Completion Endpoints
├── /logic                  ← Agenten, Tools, KI-Logik
│   ├── __init__.py
│   └── agent_controller.py ← Plan-Reason-Respond Agent
├── /models                 ← Pydantic Models
│   ├── __init__.py
│   └── chat.py             ← Chat-bezogene Models
├── /services               ← Externe Service-Integrationen
│   ├── __init__.py
│   ├── memory_qdrant.py    ← Qdrant Vector Database Service
│   └── /llms               ← LLM Provider (neue Struktur)
│       ├── __init__.py
│       ├── base_llm_provider.py      ← Abstrakte Basisklasse
│       ├── azure_openai_provider.py  ← Azure OpenAI Provider
│       ├── claude_provider.py        ← Anthropic Claude Provider
│       └── echo_provider.py          ← Echo Provider (Demo)
└── /utils                  ← Hilfsfunktionen
    ├── __init__.py
    └── logging.py          ← Logging-Konfiguration
```

## 🤖 Agent-Mode Features

Der **agent-mode** implementiert ein fortgeschrittenes "plan – reason – respond"-Paradigma:

1. **🎯 Plan**: Bestimmt das Ziel basierend auf der Benutzereingabe
2. **🧠 Reason**: Generiert Chain-of-Thought-Reasoning
3. **💬 Respond**: Erstellt finale Antwort basierend auf Planung und Überlegung

## 🚀 Verfügbare Modelle

- **`agent-mode`** - Fortgeschrittener Agent mit plan-reason-respond Paradigma
- **`gpt4.1-chat`** - Azure OpenAI GPT-4 (falls konfiguriert)
- **`claude-4-sonnet`** - Anthropic Claude (falls konfiguriert)
- **`echo-model`** - Einfaches Echo-Modell zum Testen

## 🔧 Installation und Setup

### 1. Environment aktivieren
```bash
conda activate twinly
```

### 2. Dependencies installieren
```bash
pip install -r requirements.txt
```

### 3. Environment-Variablen konfigurieren
```bash
# Azure OpenAI (optional)
export AZURE_OPENAI_ENDPOINT="your_endpoint"
export AZURE_OPENAI_API_KEY="your_key"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"
export AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4"

# Claude/Anthropic (optional)
export CLAUDE_API_KEY="your_claude_key"

# Logging Level
export LOG_LEVEL="INFO"
```

### 4. Anwendung starten
```bash
# WICHTIG: Immer vom Projekt-Root-Verzeichnis ausführen!
cd /pfad/zu/twinly
conda activate twinly
uvicorn app.main:app --reload
```

## 🔧 Lokale Entwicklung

### Wichtige Hinweise für die lokale Entwicklung

**⚠️ Häufiger Fehler vermeiden:**
- Führe uvicorn **NIEMALS** aus dem `/app` Verzeichnis aus
- Das führt zu `ModuleNotFoundError: No module named 'app'`

**✅ Korrekte Vorgehensweise:**
```bash
# 1. Zum Projekt-Root navigieren
cd /path/to/twinly

# 2. Conda Environment aktivieren
conda activate twinly

# 3. Server starten (WICHTIG: app.main:app als Modul-Pfad verwenden)
uvicorn app.main:app --reload
```

**Warum dieser Ansatz?**
- Die Anwendung verwendet absolute Imports (`from app.api.chat import ...`)
- Python muss das `app` Package vom Root-Verzeichnis aus finden können
- Der Modul-Pfad `app.main:app` zeigt auf die FastAPI-Instanz in `app/main.py`

**Entwicklung Workflow:**
1. **Terminal 1**: Server starten mit obigem Befehl
2. **Terminal 2**: Tests ausführen, Dependencies installieren, etc.
3. **Code ändern**: Auto-reload funktioniert automatisch
4. **API testen**: http://127.0.0.1:8000 oder http://127.0.0.1:8000/docs

**Debugging:**
- Server läuft auf: http://127.0.0.1:8000
- API Dokumentation: http://127.0.0.1:8000/docs
- Health Check: http://127.0.0.1:8000/health
- Logs werden in der Konsole angezeigt

## 🧪 Tests ausführen

```bash
# Alle Tests
python -m pytest -v

# Nur Main-API Tests
python -m pytest test_main.py -v

# Nur Agent Tests
python -m pytest test_agent.py -v
```

## 📡 API Endpoints

### GET `/v1/models`
Liste verfügbare Modelle auf

```bash
curl -X GET "http://127.0.0.1:8000/v1/models"
```

### POST `/v1/chat/completions`
Erstelle Chat Completions (OpenAI kompatibel)

```bash
curl -X POST "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "agent-mode",
    "messages": [
      {"role": "user", "content": "Wie lerne ich Python?"}
    ],
    "stream": false
  }'
```

### Streaming Support
```bash
curl -X POST "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "echo-model",
    "messages": [
      {"role": "user", "content": "Test streaming"}
    ],
    "stream": true
  }'
```

**Hinweis**: Streaming wird für `agent-mode` nicht unterstützt.

### GET `/health`
Health Check Endpoint

### GET `/`
Informative Homepage mit API-Dokumentation

### GET `/docs`
Automatische Swagger UI Dokumentation

## 🔍 Debugging und Logging

Die Anwendung verwendet strukturiertes Logging:

```python
from app.utils.logging import get_logger

logger = get_logger(__name__)
logger.info("Your log message")
```

Log-Level kann über die `LOG_LEVEL` Environment-Variable gesteuert werden:
- `DEBUG`
- `INFO` (Standard)
- `WARNING`
- `ERROR`
- `CRITICAL`

## 🏗️ Entwicklung

### Neue Agents hinzufügen
1. Erstelle eine neue Datei in `app/logic/`
2. Implementiere deine Agent-Logik
3. Registriere den Agent in `app/api/chat.py`
4. Füge das Modell zur Models-Liste hinzu

### Neue Services hinzufügen
1. Erstelle eine neue Service-Klasse in `app/services/`
2. Implementiere die erforderlichen Methoden
3. Integriere den Service in den entsprechenden Router

### Neue LLM Provider hinzufügen
1. Erstelle eine neue Provider-Klasse in `app/services/llms/` die von `BaseLLMProvider` erbt
2. Implementiere alle abstrakten Methoden (`is_available`, `generate_response`, `generate_streaming_response`)
3. Erstelle eine globale Instanz am Ende der Provider-Datei
4. Aktualisiere das `MODEL_MAPPING` in `app/api/chat.py`
5. Füge das Modell zur Models-Liste im `/v1/models` Endpoint hinzu
6. Schreibe Contract- und Mock-Tests für den Provider in `tests/`

### Neue Models hinzufügen
1. Definiere Pydantic Models in `app/models/`
2. Verwende Type Hints für alle Funktionen
3. Validiere Ein- und Ausgaben

## ⚡ Performance

- Async/Await für alle I/O-Operationen
- Strukturierte Logging für besseres Debugging
- Modulare Architektur für einfache Wartung
- OpenAI-kompatible API für nahtlose Integration

## 🔒 Sicherheit

- Keine sensiblen Daten in Code committen
- Environment-Variablen für API-Schlüssel verwenden
- Input-Validation über Pydantic Models
- Error Handling mit detailliertem Logging

## 📝 Konventionen

- Verwende nur `FastAPI` und `uvicorn`
- Verwende NICHT die lokale Python-Umgebung, sondern immer `conda activate twinly`
- Halte alles so minimal wie möglich
- Verwende keine unnötigen Dependencies
- Schreibe klaren, gut kommentierten Code
- API-Antworten sind immer im JSON-Format
- Schreibe Tests für alles vor der Implementierung neuer Features
- Verwende Type Hints für alle Funktions-Signaturen
- Verwende `pydantic` Models für Request- und Response-Validation
- Verwende `pytest` für Tests

## 🚫 Vermeiden

- Keine komplexen Features wie Authentifizierung oder Datenbanken
- Keine externen Tools oder Frameworks außer den genannten

## How to add a new LLM provider

Alle LLM-Provider nutzen jetzt eine einheitliche Architektur basierend auf der abstrakten `BaseLLMProvider`-Klasse:

### Schritte:
1. **Provider-Klasse erstellen**: Erstelle eine neue Datei in `app/services/llms/my_provider.py`
2. **Von BaseLLMProvider erben**: Implementiere alle abstrakten Methoden:
   - `is_available() -> bool`
   - `async generate_response(req: ChatCompletionRequest) -> Dict[str, Any]`  
   - `async generate_streaming_response(req: ChatCompletionRequest) -> AsyncGenerator[str, None]`
3. **Globale Instanz**: Erstelle eine globale Instanz am Ende der Datei: `my_provider = MyProvider()`
4. **Export hinzufügen**: Füge den Provider zu `app/services/llms/__init__.py` hinzu
5. **Model Mapping**: Aktualisiere `MODEL_MAPPING` in `app/api/chat.py`
6. **Models Endpoint**: Füge das Modell zum `/v1/models` Endpoint hinzu
7. **Tests schreiben**: Erstelle Contract- und Mock-Tests in `tests/`
8. **Linting & Type Checks**: Stelle sicher, dass `ruff`, `black` und `mypy` bestehen
9. **Dokumentation**: Dokumentiere erforderliche Environment-Variablen

### Beispiel:
```python
# app/services/llms/my_provider.py
from .base_llm_provider import BaseLLMProvider

class MyProvider(BaseLLMProvider):
    def is_available(self) -> bool:
        return True  # Check API key, etc.
    
    async def generate_response(self, req: ChatCompletionRequest):
        # Implementation here
        pass
    
    async def generate_streaming_response(self, req: ChatCompletionRequest):
        # Implementation here  
        yield "data: chunk\n\n"

# Global instance
my_provider = MyProvider()
```

### Vorteile der neuen Architektur:
- **Einheitliche API**: Alle Provider implementieren dieselben Methoden
- **Austauschbarkeit**: Provider können einfach ausgetauscht werden
- **Testbarkeit**: Jeder Provider kann isoliert getestet werden
- **Erweiterbarkeit**: Neue Provider sind einfach hinzuzufügen
- **Factory Pattern**: `MODEL_MAPPING` ermöglicht dynamische Provider-Auswahl
