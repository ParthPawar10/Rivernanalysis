# Water Quality App - Deployment Guide

## Overview
This guide will help you deploy the Water Quality web application with:
- **Backend**: FastAPI server on Render (free tier)
- **Frontend**: React app on Netlify (free tier)

---

## Part 1: Deploy Backend (FastAPI) to Render

### Step 1: Prepare Backend
The backend files are ready in the `backend/` folder with:
- ✅ `Procfile` - Tells Render how to start the app
- ✅ `requirements.txt` - Python dependencies
- ✅ `runtime.txt` - Python version

### Step 2: Deploy to Render

1. **Create Render Account**
   - Go to https://render.com
   - Sign up with GitHub/Email

2. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository OR upload the `backend/` folder

3. **Configure Service**
   - **Name**: `water-quality-api` (or your choice)
   - **Environment**: `Python 3`
   - **Region**: Choose closest to your users
   - **Branch**: `main` (or your branch name)
   - **Root Directory**: `backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

4. **Environment Variables**
   - No additional variables needed for basic deployment

5. **Create Web Service**
   - Click "Create Web Service"
   - Wait 2-3 minutes for deployment
   - Copy your backend URL (e.g., `https://water-quality-api.onrender.com`)

---

## Part 2: Deploy Frontend (React) to Netlify

### Step 1: Update Backend URL

1. Open `web/.env.production`
2. Replace `YOUR_BACKEND_URL_HERE` with your Render backend URL:
   ```
   REACT_APP_API_BASE=https://water-quality-api.onrender.com
   ```

### Step 2: Deploy to Netlify

**Option A: Netlify CLI (Recommended)**

1. Install Netlify CLI:
   ```bash
   npm install -g netlify-cli
   ```

2. Login to Netlify:
   ```bash
   netlify login
   ```

3. Deploy from the `web/` directory:
   ```bash
   cd web
   netlify deploy --prod
   ```

4. Follow prompts:
   - Create & configure new site: `Yes`
   - Team: Select your team
   - Site name: Choose a unique name (e.g., `pune-water-quality`)
   - Publish directory: `build`

**Option B: Netlify Web UI**

1. **Build the app locally**:
   ```bash
   cd web
   npm run build
   ```

2. **Create Netlify Account**
   - Go to https://netlify.com
   - Sign up with GitHub/Email

3. **Deploy via Drag & Drop**
   - Click "Sites" → "Add new site" → "Deploy manually"
   - Drag the `web/build` folder to the upload area
   - Wait for deployment

4. **Or Connect GitHub**
   - Click "Add new site" → "Import from Git"
   - Connect GitHub repository
   - Configure:
     - Base directory: `web`
     - Build command: `npm run build`
     - Publish directory: `web/build`
   - Click "Deploy site"

### Step 3: Configure Custom Domain (Optional)

1. In Netlify dashboard, go to "Domain settings"
2. Add custom domain or use the free `.netlify.app` domain

---

## Part 3: Verify Deployment

1. **Test Backend**
   - Visit: `https://your-backend-url.onrender.com/docs`
   - You should see FastAPI Swagger documentation
   - Test endpoint: `https://your-backend-url.onrender.com/predict_all?month=3&year=2024`

2. **Test Frontend**
   - Visit your Netlify URL: `https://your-site.netlify.app`
   - Click on stations to see predictions
   - Test all three modes: Predict, Interpolate, Seasonal

---

## Alternative: Deploy to Azure

If you prefer Azure:

### Backend (Azure App Service)
```bash
cd backend
az webapp up --name water-quality-api --runtime "PYTHON:3.12" --sku B1
```

### Frontend (Azure Static Web Apps)
```bash
cd web
npm run build
az staticwebapp create --name water-quality-web --source ./build
```

---

## Troubleshooting

### Backend Issues
- **"Application failed to start"**: Check Render logs
- **CORS errors**: Backend already has CORS enabled for all origins
- **Module not found**: Ensure all dependencies are in `requirements.txt`

### Frontend Issues
- **API calls failing**: Check `.env.production` has correct backend URL
- **404 on refresh**: Netlify's `netlify.toml` handles this with redirects
- **Blank page**: Check browser console for errors

### Free Tier Limitations
- **Render**: Backend may sleep after 15 min of inactivity (first request takes ~30s to wake)
- **Netlify**: 100GB bandwidth/month, 300 build minutes/month

---

## Environment Variables Summary

### Backend (Render)
No environment variables required - CORS is set to allow all origins.

### Frontend (Netlify)
Create in Netlify UI under "Site settings" → "Environment variables":
- `REACT_APP_API_BASE` = Your Render backend URL

---

## Cost Summary

- **Render (Backend)**: FREE tier
  - 512 MB RAM, Shared CPU
  - Sleeps after inactivity
  - 750 hours/month

- **Netlify (Frontend)**: FREE tier
  - 100GB bandwidth/month
  - 300 build minutes/month
  - Automatic HTTPS

**Total Cost: $0/month** 🎉

---

## Next Steps

1. ✅ Deploy backend to Render
2. ✅ Copy backend URL
3. ✅ Update `.env.production` with backend URL
4. ✅ Deploy frontend to Netlify
5. ✅ Test the live application
6. 🎊 Share your app with the world!

---

## Support

If you encounter issues:
1. Check Render logs: Dashboard → Your service → Logs
2. Check Netlify logs: Dashboard → Deploys → Deploy log
3. Check browser console for frontend errors (F12)

Good luck with your deployment! 🚀
