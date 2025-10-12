# API Verve Integration - Hate Speech Detection

## 🎯 Overview

This document describes the successful integration of **API Verve** into our hybrid hate speech detection system. API Verve provides real-time profanity filtering capabilities that enhance our detection accuracy.

## 🔧 Implementation Details

### **API Configuration**
- **Endpoint**: `https://api.apiverve.com/v1/profanityfilter`
- **Method**: POST
- **Authentication**: Bearer Token
- **Response Format**: JSON

### **Request Format**
```json
{
  "text": "Text to analyze",
  "mask": "*"
}
```

### **Response Format**
```json
{
  "status": "ok",
  "error": null,
  "data": {
    "isProfane": true,
    "filteredText": "Today is so **** hot! Why the **** would anyone go outside?",
    "mask": "*",
    "trimmed": false,
    "profaneWords": 2
  }
}
```

## 🏗️ System Architecture

### **Hybrid Detection Levels**
1. **Level 1**: API Verve (Primary - Real-time profanity detection)
2. **Level 2**: Neutrino API (Secondary - Not configured)
3. **Level 3**: API Ninja (Secondary - Not configured)
4. **Level 4**: Google Perspective API (Secondary - Not configured)
5. **Level 5**: Rule-based detection (Fallback)
6. **Level 6**: ML Model (Final fallback)

### **API Verve Integration**
- **File**: `backend/utils/api_verve.py`
- **Class**: `VerveAPIDetector`
- **Key Features**:
  - Real-time profanity detection
  - Confidence scoring based on profane word count
  - Automatic fallback to next level if unavailable
  - Error handling and timeout management

## 📊 Test Results

### **Test Cases**
| Text | Expected | Actual | Confidence | Status |
|------|----------|--------|------------|--------|
| "Hello, how are you?" | Neither | Neither | 0.10 | ✅ |
| "You are an asshole" | Offensive Language | Offensive Language | 0.70 | ✅ |
| "Fuck you bitch" | Offensive Language | Offensive Language | 0.90 | ✅ |
| "This is fucking stupid" | Offensive Language | Offensive Language | 0.70 | ✅ |
| "Good morning everyone" | Neither | Neither | 0.10 | ✅ |
| "You are a fucking idiot" | Offensive Language | Offensive Language | 0.70 | ✅ |
| "I hate you so much" | Neither | Neither | 0.10 | ✅ |
| "What a beautiful day" | Neither | Neither | 0.10 | ✅ |

### **Performance Metrics**
- **Success Rate**: 100%
- **Average Response Time**: < 1 second
- **Accuracy**: High (correctly identifies profane vs. non-profane content)
- **Confidence Scoring**: Dynamic based on profane word count

## 🔑 Configuration

### **Environment Variables**
```bash
# API Verve - Hate Speech Detection
VERVE_API_KEY=your_api_key_here
```

### **API Key Setup**
1. Register at [API Verve](https://api.apiverve.com/)
2. Get your API key from the dashboard
3. Add it to your `.env` file
4. Restart the application

## 🚀 Usage

### **In Streamlit App**
The API Verve integration is automatically used as the first level of detection in the hybrid system. No additional configuration is needed.

### **Standalone Testing**
```python
from backend.utils.api_verve import verve_detector

# Test a text
result = verve_detector.detect_hate_speech("Your text here")
print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidence']}")
```

## 🛠️ Technical Implementation

### **Key Methods**
- `detect_hate_speech(text)`: Main detection method
- `is_available()`: Check if API key is configured
- `_process_verve_response(data)`: Process API response

### **Error Handling**
- API timeout handling
- Network error management
- Invalid response processing
- Graceful fallback to next detection level

## 📈 Benefits

1. **Real-time Detection**: Immediate profanity filtering
2. **High Accuracy**: Specialized in profanity detection
3. **Confidence Scoring**: Dynamic confidence based on word count
4. **Reliability**: Robust error handling and fallback
5. **Integration**: Seamless integration with existing system

## 🔄 Next Steps

1. **Monitor Performance**: Track API usage and response times
2. **Configure Additional APIs**: Set up Neutrino, API Ninja, and Google Perspective
3. **Optimize Thresholds**: Fine-tune confidence thresholds
4. **Add Logging**: Implement detailed logging for monitoring

## 📝 Files Modified

- `backend/utils/api_verve.py` - Main API integration
- `backend/utils/__init__.py` - Import management
- `app.py` - Hybrid system integration
- `test_verve_real.py` - Testing script
- `test_hybrid_system.py` - System testing

## ✅ Status

**COMPLETED** - API Verve successfully integrated and tested. The hybrid detection system now includes real-time profanity filtering as the primary detection method.
