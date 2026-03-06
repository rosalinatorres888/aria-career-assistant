# 🚀 ARIA Deployment & Testing Guide

## Quick Test (5 minutes)

### 1. Basic Functionality Test
```bash
cd ~/Desktop/aria-career-assistant
python test_aria.py
```

This tests:
- Module imports
- Task creation
- Alert system
- Basic structure

## Local Development Setup (30 minutes)

### 1. Install Required Services

#### Redis (for task coordination)
```bash
# Mac
brew install redis
brew services start redis

# Or run temporarily
redis-server
```

#### PostgreSQL (optional, can use SQLite)
```bash
# Mac
brew install postgresql
brew services start postgresql
```

### 2. Set Up Environment Variables
```bash
# Create .env file
cat > .env << EOF
OPENAI_API_KEY=your-actual-key-here
ANTHROPIC_API_KEY=your-actual-key-here
DATABASE_URL=sqlite:///aria.db
REDIS_URL=redis://localhost:6379
EOF
```

### 3. Install Python Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Run ARIA in Test Mode
```bash
# Test with limited functionality
python src/aria.py
```

## Production Deployment Options

### Option 1: Cloud Deployment (Recommended)

#### Heroku (Free tier available)
```bash
# Install Heroku CLI
brew tap heroku/brew && brew install heroku

# Create app
heroku create aria-assistant

# Add Redis
heroku addons:create heroku-redis:hobby-dev

# Deploy
git push heroku main
```

#### Railway.app (Simple)
1. Go to https://railway.app
2. Connect GitHub repo
3. Add Redis service
4. Deploy automatically

### Option 2: VPS Deployment

#### DigitalOcean/Linode ($5/month)
```bash
# SSH to server
ssh root@your-server-ip

# Clone repo
git clone https://github.com/rosalinatorres888/aria-career-assistant

# Install requirements
apt update
apt install python3-pip redis-server
pip3 install -r requirements.txt

# Run with systemd
# Create service file...
```

### Option 3: Local 24/7 (Raspberry Pi)
- Perfect for always-on monitoring
- Low power consumption
- Can run from home network

## Testing Core Features

### 1. Test LinkedIn Integration
```python
# In Python console
from src.aria import ARIA
aria = ARIA('config/test_config.yaml')
await aria.sync_linkedin()
```

### 2. Test Opportunity Scanning
```python
await aria.scan_opportunities()
```

### 3. Test Alert System
```python
from src.aria import Alert, AlertChannel
alert = Alert(
    level="info",
    title="Test",
    message="ARIA is working!",
    channels=[AlertChannel.EMAIL]
)
await aria.send_alert(alert, AlertChannel.EMAIL)
```

## Monitoring Dashboard (Optional)

### Simple Web Interface
```python
# Create dashboard.py
import streamlit as st
import redis
import json

st.title("ARIA Control Center")

# Show task status
r = redis.Redis(host='localhost', port=6379)
tasks = r.hgetall('task_status')

st.subheader("Active Tasks")
for task, status in tasks.items():
    st.write(f"- {task}: {status}")

st.subheader("Recent Opportunities")
# Display opportunities...
```

Run with:
```bash
streamlit run dashboard.py
```

## Troubleshooting

### Common Issues:

1. **Redis Connection Error**
   - Solution: Start Redis with `redis-server`

2. **API Key Issues**
   - Solution: Check .env file has correct keys

3. **Import Errors**
   - Solution: Ensure you're in venv and installed requirements

4. **Memory Issues**
   - Solution: Use SQLite instead of PostgreSQL for testing

## Performance Tips

- Start with 1-2 monitoring tasks
- Test each integration separately
- Use mock data for initial testing
- Monitor resource usage

## Security Checklist

- [ ] API keys in .env (never commit)
- [ ] Use environment variables
- [ ] Limit API rate calls
- [ ] Secure database connection
- [ ] Enable 2FA on all platforms

## Next Steps

1. **Today**: Run test_aria.py to verify setup
2. **Tomorrow**: Set up Redis and test real monitoring
3. **This Week**: Deploy to cloud for 24/7 operation
4. **Next Week**: Add custom job boards and advanced features

---

**Support**: Create issue on GitHub if you encounter problems
**Monitoring**: Check logs in `logs/aria.log`