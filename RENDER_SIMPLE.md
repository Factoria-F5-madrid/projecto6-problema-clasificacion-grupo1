# 🚀 Render Deployment - Simple Configuration

## ⚠️ **GitHub Organization Restrictions**

**Problem**: `Factoria-F5-madrid` organization has OAuth App access restrictions.

**Solution**: Use Render with simple Python configuration.

## 🎯 **Render Configuration (Simple)**

### **Service Type:**
- ✅ **Web Service** (NOT Static Site)

### **Settings:**
| Field | Value |
|-------|-------|
| **Environment** | `Python` |
| **Root Directory** | *(leave empty)* |
| **Build Command** | `pip install -r requirements.txt` |
| **Publish Directory** | `.` |
| **Start Command** | `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0` |

## 📁 **Files Ready for Deployment**

- `app.py` - Main Streamlit application
- `hatespeech.png` - Banner image
- `backend/data/processed/cleaned_tweets.csv` - Dataset
- `requirements.txt` - Dependencies
- `.streamlit/config.toml` - Streamlit configuration

## ⏱️ **Deployment Time**

- **First time**: 5-15 minutes (normal)
- **Updates**: 2-5 minutes
- **Reason**: Render installs all dependencies from scratch

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
