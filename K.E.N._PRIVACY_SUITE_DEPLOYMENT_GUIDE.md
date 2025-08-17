# K.E.N. Privacy Suite v1.0 - Complete Deployment Guide

## 🛡️ **System Overview**

The K.E.N. Privacy Suite is a comprehensive anonymity and privacy system providing:

- **Secure OS Integration**: Qubes/Tails VM orchestration
- **Anonymous Browser Automation**: Tor-enabled browsing with anti-fingerprinting
- **Identity Generation**: Complete pseudonym profiles with ProtonMail, Privacy.com, AI Passport
- **Agentic Account Creation**: Automated account creation across platforms
- **Cloudflare Bypass**: Advanced anti-bot detection bypass with TLS spoofing
- **Credential Management**: Vaultwarden & 2FAuth integration
- **Cost Optimization**: €75/month budget with intelligent resource allocation

## 🚀 **Quick Start (1-Line Deployment)**

```bash
curl -sSL https://raw.githubusercontent.com/your-repo/knowledge-evolution-nexus/main/deploy.sh | bash
```

## 📋 **Prerequisites**

### **System Requirements**
- Ubuntu 22.04 LTS (recommended)
- 16GB RAM minimum (32GB recommended)
- 100GB SSD storage
- Docker & Docker Compose
- KVM virtualization support

### **Service Accounts Required**
- **2Captcha API Key** ($30/month budget)
- **Privacy.com Account** (virtual cards)
- **ProtonMail Account** (email generation)
- **Residential Proxy Service** ($20/month budget)

## 🔧 **Manual Installation**

### **Step 1: Clone Repository**
```bash
git clone https://github.com/your-repo/knowledge-evolution-nexus.git
cd knowledge-evolution-nexus/services/privacy-suite
```

### **Step 2: Environment Configuration**
```bash
cp .env.example .env
nano .env
```

**Required Environment Variables:**
```env
# Core Configuration
VAULTWARDEN_ADMIN_TOKEN=your_secure_admin_token
POSTGRES_PASSWORD=your_secure_db_password
GRAFANA_PASSWORD=your_secure_grafana_password

# Cloudflare Bypass
TWOCAPTCHA_API_KEY=your_2captcha_key
ANTICAPTCHA_API_KEY=your_anticaptcha_key
CAPMONSTER_API_KEY=your_capmonster_key

# Google Integration (Optional)
GOOGLE_EMAIL=your_google_account@gmail.com
GOOGLE_PASSWORD=your_google_password

# Budget Limits
SOLVING_BUDGET_MONTHLY=30.00
PROXY_BUDGET_MONTHLY=20.00
TOTAL_PRIVACY_BUDGET=75.00
```

### **Step 3: Deploy with Docker**
```bash
# Build and start all services
docker-compose up -d

# Verify deployment
docker-compose ps
```

### **Step 4: Initialize Services**
```bash
# Initialize Vaultwarden
curl -X POST http://localhost:80/admin/config \
  -H "Authorization: Bearer $VAULTWARDEN_ADMIN_TOKEN" \
  -d '{"signups_allowed": true}'

# Initialize 2FAuth
curl -X POST http://localhost:8000/api/v1/setup \
  -d '{"app_name": "K.E.N.2FAuth"}'

# Test Privacy Suite API
curl http://localhost:8080/api/v1/health
```

## 🏗️ **Architecture Components**

### **Core Services**
- **K.E.N. Privacy API** (Port 8080) - Main orchestration API
- **Vaultwarden** (Port 80) - Password & credential management
- **2FAuth** (Port 8000) - 2FA token management
- **Tor Proxy** (Port 9050) - Anonymous networking
- **PostgreSQL** (Port 5432) - Persistent data storage
- **Redis** (Port 6379) - Session & cache management

### **Browser Automation**
- **Selenium Hub** (Port 4444) - Browser orchestration
- **Firefox Nodes** - Tor-enabled Firefox instances
- **Chrome Nodes** - Undetected Chrome instances

### **Monitoring Stack**
- **Prometheus** (Port 9090) - Metrics collection
- **Grafana** (Port 3000) - Dashboard & visualization

## 🔐 **Security Configuration**

### **Network Security**
```bash
# Configure firewall
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # Vaultwarden
ufw allow 8080/tcp  # Privacy API
ufw allow 3000/tcp  # Grafana
ufw deny 5432/tcp   # PostgreSQL (internal only)
ufw deny 6379/tcp   # Redis (internal only)
ufw enable
```

### **SSL/TLS Setup**
```bash
# Generate SSL certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout privacy-suite.key -out privacy-suite.crt

# Configure reverse proxy (optional)
nginx -t && systemctl reload nginx
```

### **Credential Security**
- All passwords stored in Vaultwarden
- API keys encrypted at rest
- Biometric profiles local-only storage
- Session tokens with 1-hour expiry

## 🎯 **API Usage Examples**

### **Generate Complete Identity**
```python
import requests

response = requests.post('http://localhost:8080/api/v1/identity/generate', json={
    "locale": "en_US",
    "country": "US",
    "enable_protonmail": True,
    "enable_privacy_card": True,
    "enable_ai_passport": True
})

identity = response.json()
print(f"Generated identity: {identity['name']}")
```

### **Create Account with Privacy**
```python
response = requests.post('http://localhost:8080/api/v1/account/create', json={
    "platform": "twitter",
    "identity_id": identity['identity_id'],
    "use_existing_identity": True
})

result = response.json()
print(f"Account creation: {result['success']}")
```

### **Bypass Cloudflare Protection**
```python
response = requests.post('http://localhost:8080/api/v1/cloudflare/bypass', json={
    "url": "https://protected-site.com",
    "method": "auto"
})

bypass_result = response.json()
print(f"Bypass success: {bypass_result['success']}")
```

### **Bulk Account Creation**
```python
response = requests.post('http://localhost:8080/api/v1/account/bulk-create', json={
    "platforms": ["twitter", "reddit", "github"],
    "identity_count": 1,
    "use_same_identity": False
})

results = response.json()
for result in results['results']:
    print(f"{result['platform']}: {result['success']}")
```

## 🔍 **Monitoring & Analytics**

### **Grafana Dashboards**
Access: `http://localhost:3000` (admin/admin)

**Available Dashboards:**
- Privacy Suite Performance
- Account Creation Success Rates
- Cloudflare Bypass Statistics
- Cost Tracking & Budget Monitoring
- Identity Generation Metrics

### **Prometheus Metrics**
Access: `http://localhost:9090`

**Key Metrics:**
- `privacy_suite_accounts_created_total`
- `privacy_suite_bypass_success_rate`
- `privacy_suite_identity_generation_time`
- `privacy_suite_monthly_cost_usd`

### **Health Monitoring**
```bash
# Check all services
curl http://localhost:8080/api/v1/status

# Get performance statistics
curl http://localhost:8080/api/v1/account/statistics

# Monitor costs
curl http://localhost:8080/api/v1/budget/status
```

## 💰 **Cost Management**

### **Budget Allocation (€75/month)**
- **Solving Services**: €27 ($30)
- **Residential Proxies**: €18 ($20)
- **Infrastructure**: €20 (Hetzner Cloud)
- **Buffer**: €10

### **Cost Optimization Features**
- Automatic budget monitoring
- Service failover by cost efficiency
- Session caching to reduce API calls
- Intelligent proxy rotation

### **Budget Alerts**
```python
# Set up budget alerts
response = requests.post('http://localhost:8080/api/v1/budget/alerts', json={
    "monthly_limit": 75.00,
    "warning_threshold": 0.8,
    "critical_threshold": 0.95
})
```

## 🛠️ **Troubleshooting**

### **Common Issues**

**1. Tor Connection Failed**
```bash
# Restart Tor service
docker-compose restart tor-proxy

# Check Tor logs
docker logs ken-tor-proxy
```

**2. Cloudflare Bypass Failing**
```bash
# Check solving service balance
curl http://localhost:8080/api/v1/cloudflare/balance

# Rotate TLS fingerprints
curl -X POST http://localhost:8080/api/v1/cloudflare/rotate-fingerprints
```

**3. Identity Generation Slow**
```bash
# Check ProtonMail API status
curl http://localhost:8080/api/v1/identity/services/status

# Clear identity cache
curl -X DELETE http://localhost:8080/api/v1/identity/cache
```

**4. VM Creation Failed**
```bash
# Check KVM support
kvm-ok

# Verify virtualization
docker exec ken-privacy-api python -c "import libvirt; print('OK')"
```

### **Log Analysis**
```bash
# View all logs
docker-compose logs -f

# Specific service logs
docker logs ken-privacy-api
docker logs ken-vaultwarden
docker logs ken-2fauth
```

## 🔄 **Maintenance**

### **Daily Tasks**
```bash
# Backup credentials
docker exec ken-vaultwarden sqlite3 /data/db.sqlite3 ".backup /data/backup-$(date +%Y%m%d).db"

# Clean browser sessions
curl -X DELETE http://localhost:8080/api/v1/browser/cleanup

# Rotate Tor circuits
curl -X POST http://localhost:8080/api/v1/tor/new-circuit
```

### **Weekly Tasks**
```bash
# Update TLS fingerprints
curl -X POST http://localhost:8080/api/v1/cloudflare/update-fingerprints

# Clean old identities
curl -X DELETE http://localhost:8080/api/v1/identity/cleanup?days=30

# Generate cost report
curl http://localhost:8080/api/v1/budget/report/weekly
```

### **Monthly Tasks**
```bash
# Full system backup
./scripts/backup-privacy-suite.sh

# Update Docker images
docker-compose pull && docker-compose up -d

# Review and optimize costs
curl http://localhost:8080/api/v1/budget/optimize
```

## 🔒 **Security Best Practices**

### **Access Control**
- Use strong passwords for all services
- Enable 2FA on all external accounts
- Regularly rotate API keys
- Monitor access logs

### **Data Protection**
- Biometric profiles stored locally only
- Credentials encrypted in Vaultwarden
- Regular security audits
- Automatic session cleanup

### **Network Security**
- All traffic through Tor when possible
- Residential proxy rotation
- DNS over HTTPS
- VPN integration available

## 📊 **Performance Benchmarks**

### **Expected Performance**
- **Identity Generation**: 30-60 seconds
- **Account Creation**: 2-5 minutes per platform
- **Cloudflare Bypass**: <3 seconds average
- **Success Rates**: 95%+ for most platforms

### **Scaling Considerations**
- Horizontal scaling with Docker Swarm
- Load balancing for high-volume operations
- Database sharding for large identity stores
- CDN integration for global deployment

## 🆘 **Support & Documentation**

### **Additional Resources**
- [API Documentation](./API_DOCUMENTATION.md)
- [Security Guide](./SECURITY_GUIDE.md)
- [Performance Tuning](./PERFORMANCE_GUIDE.md)
- [Integration Examples](./examples/)

### **Community**
- GitHub Issues: Report bugs and feature requests
- Discord: Real-time support and discussions
- Documentation Wiki: Community-maintained guides

---

## 🎉 **Deployment Verification**

After deployment, verify all components:

```bash
# Run comprehensive health check
./scripts/health-check.sh

# Test identity generation
curl -X POST http://localhost:8080/api/v1/identity/generate

# Test account creation
curl -X POST http://localhost:8080/api/v1/account/create \
  -d '{"platform": "github", "use_existing_identity": false}'

# Test Cloudflare bypass
curl -X POST http://localhost:8080/api/v1/cloudflare/bypass \
  -d '{"url": "https://example.com"}'

# Check budget status
curl http://localhost:8080/api/v1/budget/status
```

**✅ Successful deployment indicators:**
- All health checks pass
- Identity generation completes in <60s
- Account creation succeeds on test platform
- Cloudflare bypass achieves >90% success rate
- Budget tracking shows accurate cost monitoring

---

**K.E.N. Privacy Suite v1.0** - Complete anonymity and privacy automation for the modern digital landscape.

