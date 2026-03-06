# ARIA / Rosalina Career Command Center

> ⚡ A production-grade autonomous career assistant — job scraping, semantic matching, archetype-aware resume generation, cover letter drafting, analytics, and application tracking.

![Dashboard](https://img.shields.io/badge/status-active-22c55e) ![Python](https://img.shields.io/badge/python-3.9+-blue) ![MongoDB](https://img.shields.io/badge/db-MongoDB-green) ![React](https://img.shields.io/badge/dashboard-React-cyan)

---

## What It Does

**ROSALINA Career Command Center** is a full-stack autonomous job search system built end-to-end as a production ML engineering portfolio project.

### 5-Tab Dashboard (`localhost:5100`)
| Tab | Features |
|-----|----------|
| **⟐ Pipeline** | Live job scraper (Built In Boston), confidence scoring, status tracking |
| **⚡ Resume AI** | Paste JD → semantic archetype scoring → fellowship-format .docx generation |
| **◎ Analytics** | Keyword heatmap, match score distribution, application funnel, source breakdown |
| **📋 Tracker** | Kanban-style application tracker with follow-up dates, inline notes |
| **✉ Cover Letter** | Claude AI-generated cover letters with stationery-format PDF export |

### Core Pipeline (`memory-brain-integration/`)
- **Job Scraper** — scrapes Built In Boston by role query, fetches full descriptions
- **Semantic Matcher** — `sentence-transformers` (all-MiniLM-L6-v2) scores JDs against 55 candidate skills
- **Archetype Classifier** — keyword scoring routes to: `AI_ML` | `Data_Engineering` | `Analytics` | `Governance_Ethics`
- **Resume Generator** — `python-docx` renders 2-column fellowship-format .docx with archetype-aware content selection
- **Memory Brain** — SQLite orchestration layer with agent dispatch messaging

---

## Architecture

```
aria-career-assistant/
├── aria_api.py              # FastAPI-style HTTP server (port 5100)
├── aria_dashboard.html      # React dashboard (5 tabs, served by aria_api.py)
├── aria_integrations.py     # Job scraper + MongoDB storage
├── aria_email_monitor.py    # Gmail IMAP alert scanner
├── aria_job_config.py       # Target roles, keywords, scoring weights
├── aria_refactored.py       # Main orchestrator loop
├── start_aria.sh            # Startup script (loads secrets from ~/.aria_secrets)
├── config/
│   └── aria_config.yaml.example   # Config template (copy → aria_config.yaml)
└── src/
    └── aria.py              # Core agent logic

memory-brain-integration/    # Separate pipeline (~/Documents/)
├── run_full_pipeline.py     # Main entry: scrape → match → generate
├── archetype_resume_generator.py  # Fellowship-format docx renderer
├── semantic_matcher.py      # sentence-transformers scoring
├── job_scraper_integrated.py
├── integration.py           # Memory Brain orchestration
└── resumes/                 # Generated .docx output folder
```

---

## Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/rosalinatorres888/aria-career-assistant.git
cd aria-career-assistant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Secrets
```bash
cp config/aria_config.yaml.example config/aria_config.yaml
# Edit aria_config.yaml — add your API keys (never commit this file)

# Or use the secrets file approach:
echo 'export ARIA_EMAIL_PASS="your-app-password"' >> ~/.aria_secrets
```

### 3. Start MongoDB
```bash
brew services start mongodb-community  # macOS
```

### 4. Launch Dashboard
```bash
source venv/bin/activate
python3 aria_api.py
# Open http://localhost:5100
```

### 5. Run Resume Pipeline
```bash
cd ~/Documents/memory-brain-integration
pip install sentence-transformers python-docx
python3 run_full_pipeline.py
# Enter role query when prompted
```

---

## Resume Generator — Direct Use

```python
from archetype_resume_generator import ResumeGenerator

gen = ResumeGenerator()
path = gen.generate_resume(
    job_id="anthropic_001",
    company="Anthropic",
    role="AI Safety Research Intern",
    score=0.78
)
print("Generated:", path)
```

Output: Fellowship-format 2-column `.docx` in `resumes/` folder with:
- Dark navy sidebar (`#1E2A3A`) — Education, Skills by category, Languages
- White right column — Summary, Top 3 projects (archetype-ranked), Experience grid
- Header: Name (28pt navy) → tagline → thin rule → small-caps contact row

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Scraping | `requests`, `BeautifulSoup`, Built In Boston |
| ML Matching | `sentence-transformers`, `all-MiniLM-L6-v2`, cosine similarity |
| Resume Generation | `python-docx`, archetype classification, fellowship format |
| Database | MongoDB (jobs), SQLite (Memory Brain orchestration) |
| Dashboard | React 18 (Babel CDN), served via Python HTTP server |
| AI Integration | Anthropic Claude API (cover letters, summaries) |
| Email | Gmail IMAP, SMTP alerts |

---

## Configuration

Copy `config/aria_config.yaml.example` to `config/aria_config.yaml` and fill in:

```yaml
openai:
  api_key: "sk-..."        # Optional

anthropic:
  api_key: "sk-ant-..."    # For cover letter generation

email:
  smtp_server: smtp.gmail.com
  port: 587
  username: your@gmail.com
  password: "xxxx xxxx xxxx xxxx"  # Gmail app password

redis:
  host: localhost
  port: 6379
```

⚠️ **Never commit `aria_config.yaml`** — it's in `.gitignore`

---

## Portfolio Context

Built as part of MS Data Analytics Engineering (Northeastern University, 4.0 GPA).  
Demonstrates: production ML pipelines, multi-agent orchestration, NLP, full-stack dashboard development, and agentic AI systems.

**Author:** Rosalina Torres  
**Portfolio:** [rosalina.sites.northeastern.edu](https://rosalina.sites.northeastern.edu)  
**GitHub:** [github.com/rosalinatorres888](https://github.com/rosalinatorres888)  
**LinkedIn:** [linkedin.com/in/rosalina2](https://linkedin.com/in/rosalina2)
