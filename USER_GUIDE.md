# MarketPulse AI - Complete User Guide 📚

## 🎯 **How to Use MarketPulse AI: Step-by-Step Guide**

This guide shows you exactly how to provide data, get insights, and see results from your MarketPulse AI system.

---

## 📊 **1. Data Input - Where to Provide Your Data**

### **Method 1: Using the Web Interface (Recommended)**

1. **Start the server**: `python run.py`
2. **Open**: http://localhost:8001/docs
3. **Find the "data" section** and click on `POST /api/v1/data/ingest`
4. **Click "Try it out"**
5. **Provide your sales data** in this format:

```json
{
  "data": [
    {
      "product_id": "PROD001",
      "date": "2024-01-15",
      "quantity_sold": 150,
      "revenue": 15000.0,
      "inventory_level": 500,
      "price": 100.0
    },
    {
      "product_id": "PROD002", 
      "date": "2024-01-15",
      "quantity_sold": 75,
      "revenue": 7500.0,
      "inventory_level": 200,
      "price": 100.0
    }
  ],
  "validate_data": true,
  "store_patterns": true
}
```

### **Method 2: Using Python Code**

```python
import requests

# Your sales data
sales_data = {
    "data": [
        {
            "product_id": "PROD001",
            "date": "2024-01-15",
            "quantity_sold": 150,
            "revenue": 15000.0,
            "inventory_level": 500,
            "price": 100.0
        }
    ],
    "validate_data": True,
    "store_patterns": True
}

# Send data to MarketPulse AI
response = requests.post(
    "http://localhost:8001/api/v1/data/ingest",
    json=sales_data
)

print("Data ingestion result:", response.json())
```

### **Method 3: Using curl (Command Line)**

```bash
curl -X POST "http://localhost:8001/api/v1/data/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {
        "product_id": "PROD001",
        "date": "2024-01-15",
        "quantity_sold": 150,
        "revenue": 15000.0,
        "inventory_level": 500,
        "price": 100.0
      }
    ],
    "validate_data": true,
    "store_patterns": true
  }'
```

---

## 🔍 **2. Getting AI Insights - Where to See Results**

### **Generate Insights for Your Products**

**Using Web Interface:**
1. Go to http://localhost:8001/docs
2. Find `POST /api/v1/insights/generate`
3. Click "Try it out"
4. Use this request:

```json
{
  "product_ids": ["PROD001", "PROD002"],
  "include_seasonal": true,
  "confidence_threshold": 0.8,
  "max_insights": 10
}
```

**Expected Response:**
```json
{
  "success": true,
  "data": {
    "insights": [
      {
        "insight_text": "Product PROD001 shows strong demand during weekends",
        "confidence_level": 0.85,
        "key_factors": ["day_of_week", "seasonal_pattern"],
        "supporting_evidence": "Sales increase by 40% on Saturdays"
      }
    ],
    "total_patterns_analyzed": 15,
    "insights_generated": 8
  },
  "message": "Insights generated successfully"
}
```

---

## 💡 **3. Getting Smart Recommendations**

### **Get Business Recommendations**

**Request:**
```json
{
  "product_ids": ["PROD001", "PROD002"],
  "business_context": {
    "target_margin": 0.25,
    "inventory_turnover_target": 12,
    "seasonal_events": ["Diwali", "Christmas"]
  },
  "priority_filter": "high",
  "include_compliance_check": true,
  "max_recommendations": 20
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "recommendations": [
      {
        "title": "Optimize inventory for PROD001",
        "description": "Increase stock by 25% before Diwali season",
        "priority": "high",
        "confidence_score": 0.92,
        "expected_impact": {
          "revenue_increase": 15000,
          "risk_reduction": 0.3
        },
        "compliance_status": "approved"
      }
    ],
    "total_products": 2,
    "compliance_validated": true
  }
}
```

---

## 📈 **4. Scenario Analysis - Test Different Strategies**

### **Compare Different Business Scenarios**

**Request:**
```json
{
  "base_parameters": {
    "product_id": "PROD001",
    "current_inventory": 1000,
    "demand_forecast": 200,
    "seasonal_events": ["Diwali"],
    "discount_range": [0.1, 0.3]
  },
  "scenario_types": ["optimistic", "pessimistic", "base"],
  "include_seasonal": true,
  "max_scenarios": 5
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "scenarios": [
      {
        "scenario_name": "optimistic",
        "expected_revenue": 45000,
        "risk_score": 0.2,
        "inventory_turnover": 8.5,
        "recommendations": [
          "Apply 20% discount during Diwali week",
          "Increase marketing budget by 30%"
        ]
      }
    ],
    "total_scenarios": 3,
    "best_scenario": "optimistic"
  }
}
```

---

## ⚠️ **5. Risk Assessment**

### **Check Inventory and Demand Risks**

**Request:**
```json
{
  "product_id": "PROD001",
  "current_inventory": 500,
  "assessment_type": "both",
  "include_seasonal_adjustment": true,
  "upcoming_events": ["Diwali", "Christmas"]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "product_id": "PROD001",
    "risk_assessments": {
      "overstock_risk": 0.3,
      "understock_risk": 0.7,
      "overall_risk": "medium"
    },
    "recommendations": [
      "Consider reducing inventory by 20%",
      "Monitor demand closely during Diwali"
    ],
    "seasonal_adjustments_applied": true
  }
}
```

---

## 🛡️ **6. Compliance Validation**

### **Check if Your Strategies are MRP Compliant**

**Request:**
```json
{
  "recommendation": {
    "product_id": "PROD001",
    "discount_percentage": 0.15,
    "new_price": 85.0,
    "original_price": 100.0,
    "strategy": "seasonal_discount"
  },
  "product_category": "electronics",
  "include_explanations": true
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "compliance_result": {
      "is_compliant": true,
      "confidence": 0.95,
      "violations": []
    },
    "regulatory_constraints": {
      "max_discount": 0.25,
      "mrp_compliance": "required"
    },
    "constraint_explanations": {
      "discount_limit": "Maximum 25% discount allowed for electronics"
    }
  }
}
```

---

## 📊 **7. Real-World Usage Examples**

### **Example 1: Small Electronics Store**

```python
import requests

# 1. Upload your sales data
sales_data = {
    "data": [
        {
            "product_id": "MOBILE_001",
            "date": "2024-01-15",
            "quantity_sold": 5,
            "revenue": 50000.0,
            "inventory_level": 20,
            "price": 10000.0
        },
        {
            "product_id": "LAPTOP_001", 
            "date": "2024-01-15",
            "quantity_sold": 2,
            "revenue": 100000.0,
            "inventory_level": 8,
            "price": 50000.0
        }
    ]
}

# Upload data
response = requests.post("http://localhost:8001/api/v1/data/ingest", json=sales_data)
print("Data uploaded:", response.json()["success"])

# 2. Get insights
insights_request = {
    "product_ids": ["MOBILE_001", "LAPTOP_001"],
    "include_seasonal": True,
    "confidence_threshold": 0.7
}

insights = requests.post("http://localhost:8001/api/v1/insights/generate", json=insights_request)
print("Insights:", insights.json()["data"]["insights"])

# 3. Get recommendations
rec_request = {
    "product_ids": ["MOBILE_001", "LAPTOP_001"],
    "business_context": {
        "target_margin": 0.20,
        "upcoming_festival": "Diwali"
    }
}

recommendations = requests.post("http://localhost:8001/api/v1/recommendations/generate", json=rec_request)
print("Recommendations:", recommendations.json()["data"]["recommendations"])
```

### **Example 2: Fashion Retail Chain**

```python
# Seasonal analysis for clothing store
scenario_request = {
    "base_parameters": {
        "product_id": "WINTER_JACKET_001",
        "current_inventory": 200,
        "demand_forecast": 150,
        "seasonal_events": ["Winter", "New Year"],
        "discount_range": [0.1, 0.4]
    },
    "scenario_types": ["winter_peak", "post_season", "clearance"],
    "include_seasonal": True
}

scenarios = requests.post("http://localhost:8001/api/v1/scenarios/analyze", json=scenario_request)
print("Best strategy:", scenarios.json()["data"]["scenarios"])
```

---

## 🔍 **8. Monitoring and System Health**

### **Check System Status**
```bash
# Health check
curl http://localhost:8001/health

# Detailed system status
curl http://localhost:8001/api/v1/system/status
```

### **View All Available Endpoints**
Go to: http://localhost:8001/docs

---

## 📁 **9. Data Storage and Persistence**

### **Where Your Data is Stored:**
- **Database**: `marketpulse.db` (SQLite file in your project directory)
- **Patterns**: Automatically extracted and stored for future analysis
- **Models**: AI models are continuously updated with your data
- **Insights**: Historical insights are saved for reference

### **Data Security:**
- All data is encrypted at rest
- No data is shared with third parties
- Complete audit trails are maintained
- Compliance with data protection regulations

---

## 🎯 **10. Quick Start Checklist**

- [ ] **Start server**: `python run.py`
- [ ] **Open docs**: http://localhost:8001/docs
- [ ] **Upload data**: Use `/api/v1/data/ingest` endpoint
- [ ] **Get insights**: Use `/api/v1/insights/generate` endpoint
- [ ] **Get recommendations**: Use `/api/v1/recommendations/generate` endpoint
- [ ] **Check compliance**: Use `/api/v1/compliance/validate` endpoint
- [ ] **Analyze scenarios**: Use `/api/v1/scenarios/analyze` endpoint

---

## 🆘 **Troubleshooting**

### **Common Issues:**

**"No data found" error:**
- Make sure you've uploaded data using `/api/v1/data/ingest` first

**"Low confidence" warnings:**
- Upload more historical data for better AI predictions
- Reduce `confidence_threshold` in requests

**"Compliance violation" errors:**
- Check MRP regulations for your product category
- Adjust discount percentages to comply with regulations

**Server not responding:**
- Check if server is running: `python run.py`
- Try different port: modify `run.py` to use port 8002

---

**🎉 You're now ready to use MarketPulse AI for intelligent retail decision-making!**

The system learns from your data and gets smarter over time. The more data you provide, the better insights and recommendations you'll receive.