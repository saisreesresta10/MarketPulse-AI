# MarketPulse AI - Project Structure Guide 📁

## 🎯 Quick Navigation

| What you want to do | Go to |
|---------------------|-------|
| **Run the demo** | `streamlit run app.py` |
| **Deploy to cloud** | Use `app.py` as main file |
| **Local development** | `frontend/` folder |
| **Read documentation** | `docs/` folder |
| **Configure deployment** | `deployment/` folder |
| **Run tests** | `tests/` folder |
| **See examples** | `examples/` folder |

## 📂 Folder Purposes

### **Root Level**
- `app.py` - **Main app for deployment** (copy of frontend/app.py)
- `requirements.txt` - **Main dependencies** for cloud deployment
- `README.md` - **Project overview** and quick start

### **Core Folders**
- `frontend/` - **Streamlit apps** (demo, cloud, production versions)
- `marketpulse_ai/` - **Core application** (API, components, storage)
- `tests/` - **Test suite** (unit tests, property-based tests)
- `examples/` - **Usage examples** and component demos

### **Configuration & Deployment**
- `config/` - **Configuration files** (dev requirements, test config)
- `deployment/` - **Platform-specific** deployment configurations
- `scripts/` - **Utility scripts** (run server, setup package)

### **Documentation & Data**
- `docs/` - **All documentation** (guides, manuals, references)
- `data/` - **Database files** (SQLite databases for local dev)

### **Development**
- `.kiro/` - **Kiro specifications** (feature specs, tasks)
- `.streamlit/` - **Streamlit configuration**

## 🚀 Common Commands

```bash
# Demo (local)
streamlit run app.py

# Full development setup
pip install -r config/requirements-dev.txt
python scripts/run.py  # API server
streamlit run frontend/streamlit_production_app.py  # UI

# Testing
pytest tests/

# Deployment
# Use app.py as main file on Streamlit Cloud
```

## 📋 File Locations Quick Reference

| Old Location | New Location | Purpose |
|--------------|--------------|---------|
| `app.py` | `app.py` + `frontend/app.py` | Main demo app |
| `DEPLOYMENT_GUIDE.md` | `docs/DEPLOYMENT_GUIDE.md` | Deployment instructions |
| `run.py` | `scripts/run.py` | API server startup |
| `requirements-dev.txt` | `config/requirements-dev.txt` | Dev dependencies |
| `Procfile` | `deployment/Procfile` | Heroku config |

---

**Everything is now organized and easy to find!** 🎉