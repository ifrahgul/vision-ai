# 🏥 Hospital Grade Vision Screening System

![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV Version](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![Flask Version](https://img.shields.io/badge/Flask-2.3%2B-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

A comprehensive eye vision screening system with 8 medical-grade tests, real-time eye tracking, and automatic diagnosis. Built with Python, OpenCV, and Flask for hospitals, clinics, and home use.

---

## 📋 Table of Contents
- [Features](#-features)
- [Screenshots](#-screenshots)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [How to Use](#-how-to-use)
- [Test Descriptions](#-test-descriptions)
- [Results Interpretation](#-results-interpretation)
- [Sample Output](#-sample-output)
- [Project Structure](#-project-structure)
- [Deployment Options](#-deployment-options)
- [Troubleshooting](#-troubleshooting)
- [Requirements](#-requirements)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## ✨ Features

### 8 Complete Vision Tests
1. **Myopia Test** - Near-sightedness detection using Snellen chart letters (E, H, N, O, S...)
2. **Hyperopia Test** - Far-sightedness detection using reading text passages
3. **Astigmatism Test** - Line pattern detection (horizontal, vertical, diagonal, grid)
4. **Contrast Sensitivity Test** - Gray scale pattern recognition
5. **Color Vision Test** - Ishihara-style colored number plates
6. **Visual Field Test** - Peripheral vision assessment with position dots
7. **Eye Alignment Test** - Phoria and binocular vision evaluation
8. **Accommodation Test** - Focusing ability at different distances

### Real-time Analysis
- Live camera feed with face and eye detection
- Eye movement tracking with left/right eye graphs
- Automatic response recording and scoring
- Progress tracking for each test level

### Comprehensive Reporting
- Overall vision score calculation
- Test-by-test results grid
- Automatic medical diagnosis
- Personalized recommendations
- Three visual graphs:
  - Eye Tracking Analysis
  - Test Results Bar Chart
  - Vision Health Overview Pie Chart
- Print-ready reports
- CSV and TXT export

### User Interface
- Professional hospital-grade design
- Mobile responsive layout
- Real-time pattern display
- One-click test reset
- On-screen results (no download needed)

---

## 📸 Screenshots
<img width="1836" height="815" alt="image" src="https://github.com/user-attachments/assets/7e81f8e1-0c22-4528-9834-64dd3901a9a5" />

<img width="1835" height="791" alt="image" src="https://github.com/user-attachments/assets/ebc26dd5-1aa7-405b-a7cf-30d94390d453" />

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/ifrahgul/vision-ai.git

# Navigate to project folder
cd vision-ai

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate
# Activate virtual environment (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

# Open browser and visit
http://localhost:5000



