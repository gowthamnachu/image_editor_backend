# Backend Deployment Guide

## Vercel Deployment

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

3. **Follow the prompts:**
   - Set up and deploy? Yes
   - Which scope? Select your account
   - Link to existing project? No
   - What's your project's name? (e.g., image-editor-backend)
   - In which directory is your code located? ./
   - Want to override settings? No

4. **Get your deployment URL**
   - After deployment, you'll get a URL like: `https://your-project.vercel.app`
   - Copy this URL - you'll need it for the frontend configuration

5. **Set up custom domain (optional)**
   - Go to your project settings in Vercel dashboard
   - Add your custom domain

### Important Notes

⚠️ **Vercel Limitations for this project:**
- Vercel has a 50MB deployment size limit
- OpenCV and Mediapipe libraries are large (>50MB)
- **Alternative: Deploy to Railway, Render, or Heroku instead**

### Alternative: Railway Deployment (Recommended)

1. **Install Railway CLI**
   ```bash
   npm install -g @railway/cli
   ```

2. **Login and deploy**
   ```bash
   cd backend
   railway login
   railway init
   railway up
   ```

3. **Set environment variables** (if needed)
   ```bash
   railway variables set PYTHONUNBUFFERED=1
   ```

### Alternative: Render Deployment (Recommended)

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Use these settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Environment:** Python 3

### Environment Variables
No environment variables required for basic setup.

### Testing
After deployment, test the API:
```bash
curl https://your-backend-url.com/api/health
```

## Troubleshooting

- If you get size errors with Vercel, use Railway or Render instead
- Check logs: `vercel logs` or in the platform dashboard
- Ensure all dependencies are in requirements.txt
