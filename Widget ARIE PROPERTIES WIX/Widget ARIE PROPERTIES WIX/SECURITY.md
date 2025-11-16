# 🔒 Security Guide - ARIE Properties Wix Widget

## ⚠️ CRITICAL: Protected Files

This repository **DOES NOT** include sensitive files with real credentials. The following files are protected by `.gitignore`:

### Files NOT Included in Repository:
- ✗ `backend-proxy.js` (contains Sienge API password)
- ✗ `http-functions.js` (contains authentication credentials)
- ✗ `config.js` (contains API configuration)
- ✗ `cors-proxy/server.js` (may contain secrets)
- ✗ Any `*.txt` or `*.json` files with customer data
- ✗ Node modules and dependencies

---

## 🔐 Setup Instructions

### 1. Configure Sienge API Credentials

**Create your own `backend-proxy.js`:**

```bash
# Copy the example file
cp backend-proxy.js.example backend-proxy.js
```

**Edit and replace placeholders:**

```javascript
const API_CONFIG = {
    baseURL: 'https://api.sienge.com.br/YOUR_COMPANY/public/api/v1',
    auth: {
        username: 'YOUR_USERNAME',  // ⚠️ Replace with your Sienge username
        password: 'YOUR_PASSWORD'   // ⚠️ Replace with your Sienge password
    }
};
```

**Where to get credentials:**
- Contact your Sienge ERP administrator
- Access Sienge admin panel > API Settings
- Create dedicated API user (recommended)

---

### 2. Configure Wix Integration

**Create your own `config.js`:**

```bash
# Copy the example file
cp config.js.example config.js
```

**Set up environment variables in Wix:**

1. Access Wix Editor > Dev Mode
2. Go to Backend > Secrets Manager
3. Add the following secrets:
   - `WIX_API_KEY`: Your Wix API key
   - `WIX_SITE_ID`: Your Wix site ID
   - `SIENGE_USERNAME`: Sienge API username
   - `SIENGE_PASSWORD`: Sienge API password

**Update config.js to use secrets:**

```javascript
wix: {
    apiKey: process.env.WIX_API_KEY,
    siteId: process.env.WIX_SITE_ID,
},
sienge: {
    username: process.env.SIENGE_USERNAME,
    password: process.env.SIENGE_PASSWORD
}
```

---

### 3. Secure CORS Proxy (if using Node.js proxy)

**Create `cors-proxy/server.js`:**

```bash
cd cors-proxy
cp server.js.example server.js  # If example exists
```

**Use environment variables:**

```javascript
const PORT = process.env.PORT || 3000;
const SIENGE_API_URL = process.env.SIENGE_API_URL;
const SIENGE_USER = process.env.SIENGE_USER;
const SIENGE_PASS = process.env.SIENGE_PASS;
```

**Create `.env` file for local development:**

```bash
# .env (NEVER commit this file!)
SIENGE_API_URL=https://api.sienge.com.br/YOUR_COMPANY/public/api/v1
SIENGE_USER=your_username
SIENGE_PASS=your_password
PORT=3000
```

---

## 🛡️ Security Best Practices

### ✅ DO:

1. **Use Environment Variables**
   - Store all credentials in environment variables
   - Never hardcode passwords in source code
   - Use Wix Secrets Manager for production

2. **API Key Restrictions**
   - Limit API keys to specific domains
   - Set IP whitelist if possible
   - Rotate credentials periodically

3. **Input Validation**
   - Always validate CPF/CNPJ format
   - Sanitize user inputs
   - Implement rate limiting

4. **HTTPS Only**
   - Use HTTPS for all API calls
   - Enable SSL/TLS on proxy server
   - Verify SSL certificates

5. **Error Handling**
   - Never expose stack traces to users
   - Log errors securely (backend only)
   - Return generic error messages

### ❌ DON'T:

1. **Never Commit:**
   - Real API credentials
   - Customer data (CPF/CNPJ, names, addresses)
   - Authentication tokens
   - `.env` files

2. **Never Expose:**
   - Backend credentials to frontend
   - Full error messages to users
   - Database queries in client-side code

3. **Never Use:**
   - HTTP for production
   - Default/weak passwords
   - Same credentials across environments

---

## 🔍 Credential Checklist

Before deploying, verify:

- [ ] All credentials removed from frontend code
- [ ] `.gitignore` includes all sensitive files
- [ ] Environment variables configured in Wix
- [ ] API keys restricted to production domain
- [ ] CORS properly configured
- [ ] HTTPS enabled on all endpoints
- [ ] Error logging configured (backend only)
- [ ] Rate limiting implemented
- [ ] Input validation in place
- [ ] Security headers configured

---

## 🚨 If Credentials Are Exposed

**Immediate Actions:**

1. **Regenerate Credentials:**
   - Change Sienge API password immediately
   - Rotate Wix API keys
   - Update all environment variables

2. **Review Git History:**
   ```bash
   # Check if credentials are in Git history
   git log -p | grep -i "password"
   ```

3. **Remove from Git History (if exposed):**
   ```bash
   # WARNING: Rewrites history!
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch backend-proxy.js" \
     --prune-empty --tag-name-filter cat -- --all

   git push origin --force --all
   ```

4. **Notify Stakeholders:**
   - Inform your security team
   - Contact Sienge support
   - Monitor API usage for anomalies

---

## 📞 Support

**Security Issues:**
- Contact ARIE Properties IT department
- Email: ti@arieproperties.com.br

**Sienge API Support:**
- Sienge documentation: https://api.sienge.com.br/docs
- Sienge support: suporte@sienge.com.br

**Wix Velo Support:**
- Wix Velo documentation: https://www.wix.com/velo/reference
- Wix support center

---

## 📋 Security Audit Log

Track all credential changes:

| Date | Action | User | Notes |
|------|--------|------|-------|
| 2024-07-XX | Initial setup | Dev Team | Created API credentials |
| YYYY-MM-DD | [Action] | [User] | [Details] |

---

**Last Updated:** 2024-07-16
**Security Policy Version:** 1.0
