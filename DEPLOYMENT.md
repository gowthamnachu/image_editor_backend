# Backend Deployment Guide

⚠️ **Important: Vercel has size limitations (250MB) and may not work with OpenCV + Mediapipe.**

**Recommended platforms: Render.com (Free tier) or Railway.app**

---

## Option 1: Render.com (RECOMMENDED - Free Tier Available)

### Prerequisites
- Render account (sign up at render.com)
- GitHub account with your code pushed

### Deployment Steps

1. **Push your code to GitHub**
   ```bash
   cd backend
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin YOUR_GITHUB_REPO_URL
   git push -u origin main
   ```

2. **Create Web Service on Render**
   - Go to https://dashboard.render.com/
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the backend directory (if monorepo)

3. **Configure Build Settings**
   - **Name:** image-editor-backend
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** Free

4. **Deploy**
   - Click "Create Web Service"
   - Wait 5-10 minutes for the build to complete
   - Copy your deployment URL: `https://your-app.onrender.com`

### Testing
```bash
curl https://your-app.onrender.com/api/health
```

---

## Option 2: Railway.app (RECOMMENDED - $5 credit free)

### Prerequisites
- Railway account (sign up at railway.app)
- Railway CLI installed: `npm install -g @railway/cli`

### Deployment Steps

1. **Login to Railway**
   ```bash
   railway login
   ```

2. **Initialize and deploy**
   ```bash
   cd backend
   railway init
   railway up
   ```

3. **Add domain**
   ```bash
   railway domain
   ```

4. **Set environment variables (if needed)**
   ```bash
   railway variables set PYTHONUNBUFFERED=1
   ```

### Testing
```bash
curl https://your-app.railway.app/api/health
```

---

## Option 3: Vercel (May hit size limits - NOT RECOMMENDED)

### Prerequisites
- Vercel account (sign up at vercel.com)
- Vercel CLI installed: `npm install -g vercel`

### Deployment Steps

1. **Login to Vercel**
   ```bash
   vercel login
   ```

2. **Deploy from backend directory**
   ```bash
   cd backend
   vercel
   ```

3. **Follow the prompts**

⚠️ **Warning:** Vercel has a 250MB uncompressed size limit. OpenCV and Mediapipe together exceed this. If deployment fails, use Render or Railway instead.

---

## Option 4: Heroku

### Prerequisites
- Heroku account
- Heroku CLI installed

### Deployment Steps

1. **Login to Heroku**
   ```bash
   heroku login
   ```

2. **Create app**
   ```bash
   cd backend
   heroku create your-app-name
   ```

3. **Deploy**
   ```bash
   git init
   git add .
   git commit -m "Deploy"
   git push heroku main
   ```

4. **Open app**
   ```bash
   heroku open
   ```

---

## After Backend Deployment

Once your backend is deployed, you'll get a URL like:
- Render: `https://your-app.onrender.com`
- Railway: `https://your-app.railway.app`
- Heroku: `https://your-app.herokuapp.com`

**Copy this URL** - you'll need it to configure the frontend!

---

## Troubleshooting

### Build Fails - Size Limit
**Solution:** Use Render or Railway instead of Vercel

### Import Errors
**Solution:** Ensure all files are committed and .vercelignore doesn't exclude necessary files

### CORS Errors
**Solution:** Update `allow_origins` in `app/main.py` with your frontend URL

### Cold Starts (Free Tier)
Free tier services may sleep after inactivity. First request takes 30-60 seconds.

---

## Environment Variables
No environment variables required for basic setup.

## Performance Notes
- First request may be slow (cold start on free tier)
- Render/Railway free tiers sleep after 15 minutes of inactivity
- Consider upgrading to paid tier for production use

