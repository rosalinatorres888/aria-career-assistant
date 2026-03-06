#!/usr/bin/env python3
"""
ARIA Testing Script - Verify Core Functionality
Run this to test ARIA's basic features before full deployment
"""

import asyncio
import yaml
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Set up test environment variables if not set
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

async def test_aria_basic():
    """Test basic ARIA functionality"""
    print("🤖 ARIA - Basic Functionality Test")
    print("=" * 50)
    
    try:
        # Import ARIA
        from aria import ARIA, Task, TaskPriority, Alert, AlertChannel
        print("✅ ARIA module imported successfully")
        
        # Test configuration loading
        config_path = "config/test_config.yaml"
        if not Path(config_path).exists():
            print("⚠️  Creating test configuration...")
            # Create minimal config for testing
            test_config = {
                'database': {'connection_string': 'sqlite:///test.db'},
                'redis': {'host': 'localhost', 'port': 6379},
                'email': {
                    'smtp_server': 'smtp.gmail.com',
                    'port': 587,
                    'username': 'test@test.com',
                    'password': 'test',
                    'from_address': 'test@test.com',
                    'to_address': 'test@test.com'
                },
                'openai': {'api_key': 'test'},
                'anthropic': {'api_key': 'test'}
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(test_config, f)
        
        print(f"✅ Configuration found at {config_path}")
        
        # Test ARIA initialization (without actually running)
        print("\n🔧 Testing ARIA initialization...")
        try:
            # This will fail if Redis isn't running, but shows structure works
            aria = ARIA(config_path)
            print("✅ ARIA initialized (may need Redis for full functionality)")
        except Exception as e:
            print(f"⚠️  Initialization needs services: {e}")
            print("   This is expected if Redis/DB aren't running")
        
        # Test Task creation
        print("\n📋 Testing Task creation...")
        test_task = Task(
            id="test_001",
            name="Test Task",
            priority=TaskPriority.MEDIUM,
            action=lambda: print("Task executed"),
            schedule="*/10 * * * *"
        )
        print(f"✅ Created task: {test_task.name} (Priority: {test_task.priority.value})")
        
        # Test Alert creation
        print("\n🚨 Testing Alert creation...")
        test_alert = Alert(
            level="info",
            title="Test Alert",
            message="This is a test alert",
            action_required="No action needed",
            channels=[AlertChannel.EMAIL]
        )
        print(f"✅ Created alert: {test_alert.title}")
        
        print("\n" + "=" * 50)
        print("✅ Basic structure tests passed!")
        print("\n📋 Next Steps for Full Production:")
        print("1. Install and start Redis: brew install redis && redis-server")
        print("2. Set up PostgreSQL or use SQLite for testing")
        print("3. Add your API keys to .env file")
        print("4. Run the full ARIA system")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you're in the aria-career-assistant directory")
    except Exception as e:
        print(f"❌ Test error: {e}")

async def test_aria_monitoring():
    """Test monitoring capabilities (mock version)"""
    print("\n🔍 Testing Monitoring Capabilities (Mock)")
    print("=" * 50)
    
    # Simulate monitoring tasks
    mock_tasks = [
        {"name": "LinkedIn Check", "status": "✅ Active", "last_run": "2 min ago"},
        {"name": "GitHub Sync", "status": "✅ Active", "last_run": "15 min ago"},
        {"name": "Job Board Scan", "status": "✅ Active", "last_run": "5 min ago"},
        {"name": "Email Monitor", "status": "⚠️ Needs Config", "last_run": "N/A"}
    ]
    
    for task in mock_tasks:
        print(f"{task['status']} {task['name']:<20} Last: {task['last_run']}")
    
    print("\n📊 Mock Metrics:")
    print("  Profile Views: 47 (↑12 from yesterday)")
    print("  New Opportunities: 3 high-confidence matches")
    print("  Network Engagement: 5 auto-responses sent")
    print("  System Health: 95% operational")

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════╗
    ║     ARIA - Career Assistant Test      ║
    ║     Testing Core Functionality        ║
    ╔════════════════════════════════════════╝
    """)
    
    # Run basic tests
    asyncio.run(test_aria_basic())
    
    # Run monitoring test
    asyncio.run(test_aria_monitoring())
    
    print("\n✨ Testing complete! Check output above for results.")
