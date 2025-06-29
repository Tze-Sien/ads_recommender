# Ad Recommender System 🎯

A machine learning-powered advertisement recommendation system built with Streamlit, featuring an Instagram-like gallery interface for browsing and discovering personalized ads based on user preferences.

## 📋 Features

- **Interactive Gallery**: Instagram-style interface for browsing advertisements
- **Personalized Recommendations**: ML-powered content-based filtering using sentence transformers
- **User Preferences**: Like/dislike system to improve recommendations
- **Real-time Learning**: System adapts to user behavior over time
- **Modern UI**: Clean, responsive design with smooth navigation

## 🚀 Quick Start Guide for Beginners

### Prerequisites

Before you begin, make sure you have Python installed on your system:
- Python 3.8 or higher
- pip (Python package installer)

To check if Python is installed:
```bash
python --version
# or
python3 --version
```

### Step 1: Download the Project

Clone or download this project to your local machine and navigate to the project directory:
```bash
cd Ads_Recommender
```

### Step 2: Create a Virtual Environment

A virtual environment helps keep your project dependencies isolated from other Python projects.

**On macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

**On Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

You should see `(venv)` at the beginning of your terminal prompt, indicating the virtual environment is active.

### Step 3: Install Dependencies

Install all required packages using pip:
```bash
pip install -r requirements.txt
```

This will install:
- Streamlit (web framework)
- PyTorch (machine learning framework)
- Sentence Transformers (for text embeddings)
- Pandas & NumPy (data processing)
- Scikit-learn (machine learning utilities)
- Pillow (image processing)

### Step 4: Initialize the Database

Set up the initial database and load the data:
```bash
python db-init.py
```

### Step 5: Run the Application

Start the Streamlit web application:
```bash
streamlit run app.py --server.port 8501 --server.address localhost
```

**Alternative: Use the provided script (recommended):**
```bash
# Make the script executable (macOS/Linux only)
chmod +x run_app.sh

# Run the script
./run_app.sh
```

### Step 6: Access the Application

Open your web browser and go to:
```
http://localhost:8501
```

You should see the Ad Recommender application with an Instagram-like interface!

## 🖥️ How to Use the Application

### Navigation
The application has three main sections accessible via the sidebar:

1. **🏠 Main Gallery**: Browse all available advertisements
2. **❤️ Liked Posts**: View your liked advertisements
3. **🎯 Recommendations**: Get personalized ad recommendations

### Interacting with Ads
- **Like**: Click the heart button (❤️) to like an ad
- **Unlike**: Click the heart button again to remove the like
- **View Details**: Each ad shows title, description, and category information

### Getting Recommendations
1. Browse the main gallery and like several ads that interest you
2. Navigate to the "Recommendations" page
3. The system will suggest similar ads based on your preferences
4. The more you interact, the better the recommendations become!

## 📁 Project Structure

```
Ads_Recommender/
├── app.py                 # Main Streamlit application
├── db-init.py            # Database initialization script
├── requirements.txt      # Python dependencies
├── run_app.sh           # Startup script
├── ads_recommendation.db # SQLite database (created after init)
├── components/          # UI components
│   ├── pages.py        # Page layouts and logic
│   └── ui_components.py # Reusable UI elements
├── config/             # Configuration files
│   └── gallery_config.py # App configuration
├── core/               # Core business logic
│   ├── database.py     # Database operations
│   └── recommender.py  # Recommendation algorithms
├── data/               # Training data (Parquet files)
├── utils/              # Utility functions
│   ├── data_loader.py  # Data loading utilities
│   ├── image_utils.py  # Image processing
│   └── styles.py       # CSS styling
└── vectorizer/         # Pre-trained models
    ├── tfidf_matrix.npz # TF-IDF matrix
    └── vectorizer.pkl   # Trained vectorizer
```

## 🛠️ Troubleshooting

### Common Issues

**1. "Command not found" errors:**
- Make sure Python is installed and added to your PATH
- Try using `python3` instead of `python`

**2. Permission denied on run_app.sh:**
```bash
chmod +x run_app.sh
```

**3. Port already in use:**
```bash
streamlit run app.py --server.port 8502  # Try different port
```

**4. Module import errors:**
- Ensure virtual environment is activated: `source venv/bin/activate`
- Reinstall requirements: `pip install -r requirements.txt`

**5. Database errors:**
- Delete `ads_recommendation.db` and run `python db-init.py` again

### Performance Tips

- The first run may take longer as models are being loaded
- For better performance on slower machines, the app automatically sets optimal environment variables
- Close other resource-intensive applications while running

## 🔧 Development

### Adding New Features

1. **New Pages**: Add to `components/pages.py`
2. **UI Components**: Add to `components/ui_components.py`
3. **Database Models**: Modify `core/database.py`
4. **Recommendation Logic**: Update `core/recommender.py`

### Configuration

Edit `config/gallery_config.py` to customize:
- App title and icon
- Display settings
- Gallery layout options

## 📊 Data

The application uses a curated dataset of advertisements with:
- Text descriptions
- Categories
- Visual elements
- Engagement metrics

Data is automatically loaded from the `data/` directory during initialization.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational purposes. Please respect the licensing terms of all dependencies.

## 💡 Tips for Beginners

1. **Start Simple**: Like a few ads in different categories to see how recommendations work
2. **Explore**: Try all three pages to understand the full functionality
3. **Experiment**: The system learns from your interactions, so feel free to like/unlike different content
4. **Learn**: Check out the code structure to understand how the recommendation system works

## 🆘 Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Ensure all steps were followed correctly
3. Verify your Python version and virtual environment setup
4. Check that all dependencies installed successfully

---

**Happy exploring! 🚀**

*This project demonstrates modern web development with Python, machine learning integration, and user-centric design principles.*
