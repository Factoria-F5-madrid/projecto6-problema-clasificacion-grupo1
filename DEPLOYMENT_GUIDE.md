# 🚀 Deployment Guide - Hate Speech Detection System

## ⚠️ **GitHub Organization Restrictions**

**Problem**: `Factoria-F5-madrid` organization has OAuth App access restrictions.

**Solution**: Use alternative deployment platforms that don't require GitHub OAuth.

## 🎯 **Recommended Deployment Options**

### **Option 1: Railway (BEST for your situation)**
- **Free**: Yes (with limitations)
- **No OAuth issues**: Uses GitHub tokens
- **Easy**: Just connect repository
- **URL**: `https://your-app-name.railway.app`

**Steps**:
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Click "New Project"
4. Select "Deploy from GitHub repo"
5. Choose your repository
6. Railway will auto-detect it's a Streamlit app
7. Deploy!

### **Option 2: Render**
- **Free**: Yes (with limitations)
- **No OAuth issues**: Uses GitHub tokens
- **More control**: Custom Dockerfile
- **URL**: `https://your-app-name.onrender.com`

**Steps**:
1. Go to [render.com](https://render.com)
2. Sign up with GitHub
3. Click "New +"
4. Select "Web Service"
5. Connect your repository
6. Use these settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
7. Deploy!

### **Option 3: Fork Repository (Alternative)**
- **Free**: Yes
- **Workaround**: Create personal fork
- **Steps**:
  1. Fork repository to your personal GitHub
  2. Deploy from personal fork
  3. Avoid organization restrictions

## 📁 **Files Ready for Deployment**

- `app.py` - Main Streamlit application
- `hatespeech.png` - Banner image
- `backend/data/processed/cleaned_tweets.csv` - Dataset
- `requirements.txt` - Dependencies
- `Dockerfile` - For Render deployment
- `.streamlit/config.toml` - Streamlit configuration

## 🔧 **Quick Test Locally**

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py

# Access at http://localhost:8501
```

## 📊 **Model Performance**

- **Accuracy**: 82.8%
- **F1-Score**: 84.2%
- **Overfitting**: 1.52% (< 5% required)
- **Hybrid Model**: Rule-based + ML for better detection

## 🎯 **Features**

- **Real-time Detection**: Type text, get instant results
- **Hybrid Classification**: Rules + ML for accuracy
- **3 Pages**: Detector, Data Analysis, Model Metrics
- **Responsive Design**: Works on mobile and desktop

---
*Ready for deployment! 🚀*
