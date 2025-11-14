# 🤖 ARIA - Autonomous Career Assistant

24/7 AI-powered career management system that handles everything from opportunity detection to network engagement with zero manual intervention.

## ✨ Key Features

### 🎯 Autonomous Monitoring
- **Multi-Platform Sync**: GitHub, LinkedIn, AngelList, YCombinator
- **Real-time Opportunity Detection**: Scans 10+ job boards every 15 minutes
- **Weak Signal Analysis**: Identifies emerging opportunities
- **Profile Analytics**: Tracks who's viewing your profiles

### 🤝 Intelligent Engagement  
- **Auto-Network Engagement**: Likes, comments based on relevance
- **Smart Follow-ups**: Automatically sends application follow-ups
- **Recruiter Response Drafting**: AI-generated responses
- **Content Scheduling**: Publishes at peak engagement times

### 🚨 Multi-Channel Alerts
- Email, SMS (Twilio), Slack, Discord, Telegram
- Priority-based alerting (Critical/High/Medium/Low)
- Voice call alerts for urgent opportunities

### 🧠 AI-Powered Intelligence
- **GPT-4 & Claude Integration**: Complex reasoning
- **Custom ML Models**: Opportunity ranking and matching
- **Real-time Decision Making**: Autonomous responses

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure ARIA
cp config/example_config.yaml config/aria_config.yaml
# Edit with your API keys

# Run ARIA
python src/aria.py
```

## 🏗️ Architecture

- **Async Processing**: Handle multiple tasks simultaneously
- **Redis**: Real-time coordination
- **Celery**: Distributed task processing
- **SQLAlchemy**: Data persistence

## 📊 Task Priority System

| Priority | Response Time | Use Case |
|----------|--------------|----------|
| CRITICAL | Immediate | Recruiter contact, Interview invites |
| HIGH | Within 2 hours | Application deadlines |
| MEDIUM | Within 24 hours | Follow-ups, Engagement |
| LOW | Within week | Content generation |

## 👩‍💻 Author

Rosalina Torres - ML/AI Engineer
- LinkedIn: [linkedin.com/in/rosalinatorres](https://linkedin.com/in/rosalina-torres)
- GitHub: [@rosalinatorres888](https://github.com/rosalinatorres888)

## 📄 License

MIT License
