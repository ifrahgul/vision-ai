import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request
import threading
import time
from datetime import datetime
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import io
import base64
import socket

app = Flask(__name__)


class VisionScreeningSystem:
    def __init__(self):
        print("🔍 Initializing Hospital Grade Vision Screening System...")
        
        # Load Haar Cascades
        cascade_path = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(
            cascade_path + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cascade_path + 'haarcascade_eye.xml'
        )
        
        # Create directories
        self.results_dir = "vision_screening_results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        
        # ==========================================
        # COMPREHENSIVE VISION TESTS WITH PROPER SHAPES/TEXT
        # ==========================================
        
        # 1. MYOPIA TEST (Distance Vision) - Letters
        self.myopia_tests = {
            'name': 'Myopia Test (Distance Vision)',
            'type': 'myopia',
            'levels': [
                {'value': 'E', 'size': 100, 'desc': 'Letter E - 6/60'},
                {'value': 'H', 'size': 80, 'desc': 'Letter H - 6/36'},
                {'value': 'N', 'size': 60, 'desc': 'Letter N - 6/24'},
                {'value': 'O', 'size': 50, 'desc': 'Letter O - 6/18'},
                {'value': 'S', 'size': 40, 'desc': 'Letter S - 6/12'},
                {'value': 'V', 'size': 30, 'desc': 'Letter V - 6/9'},
                {'value': 'Z', 'size': 25, 'desc': 'Letter Z - 6/7.5'},
                {'value': 'T', 'size': 20, 'desc': 'Letter T - 6/6'},
                {'value': 'L', 'size': 15, 'desc': 'Letter L - 6/5'},
                {'value': 'C', 'size': 12, 'desc': 'Letter C - 6/4'},
                {'value': 'F', 'size': 10, 'desc': 'Letter F - 6/3'},
                {'value': 'P', 'size': 8, 'desc': 'Letter P - 6/2.5'},
                {'value': 'D', 'size': 6, 'desc': 'Letter D - 6/2'}
            ],
            'current_index': 0
        }
        
        # 2. HYPEROPIA TEST (Near Vision) - Reading Text
        self.hyperopia_tests = {
            'name': 'Hyperopia Test (Reading Vision)',
            'type': 'hyperopia',
            'levels': [
                {'value': 'The quick brown fox jumps', 'size': 40, 'desc': 'Large Text - N5'},
                {'value': 'Pack my box with five dozen', 'size': 35, 'desc': 'Large Text - N6'},
                {'value': 'How vexingly quick daft zebras', 'size': 30, 'desc': 'Medium Text - N8'},
                {'value': 'Bright vixens jump; dozy fowl', 'size': 25, 'desc': 'Medium Text - N10'},
                {'value': 'Jinxed wizards pluck ivy', 'size': 20, 'desc': 'Small Text - N12'},
                {'value': 'Sphinx of black quartz', 'size': 18, 'desc': 'Small Text - N14'},
                {'value': 'Five quacking zephyrs', 'size': 16, 'desc': 'Very Small - N18'},
                {'value': 'Waltz bad nymph', 'size': 14, 'desc': 'Very Small - N24'},
                {'value': 'Glib jocks quiz', 'size': 12, 'desc': 'Fine Print - N36'},
                {'value': 'Quick zephyrs', 'size': 10, 'desc': 'Fine Print - N48'}
            ],
            'current_index': 0
        }
        
        # 3. ASTIGMATISM TEST - Different Line Patterns
        self.astigmatism_tests = {
            'name': 'Astigmatism Test',
            'type': 'astigmatism',
            'levels': [
                {'value': '───', 'angle': 0, 'desc': 'Horizontal Lines'},
                {'value': '│││', 'angle': 90, 'desc': 'Vertical Lines'},
                {'value': '╱╱╱', 'angle': 45, 'desc': 'Diagonal Lines (45°)'},
                {'value': '╲╲╲', 'angle': 135, 'desc': 'Diagonal Lines (135°)'},
                {'value': '┼┼┼', 'angle': 'grid', 'desc': 'Cross Grid Pattern'},
                {'value': '⬤⬤⬤', 'angle': 'dots', 'desc': 'Dot Pattern Test'}
            ],
            'current_index': 0
        }
        
        # 4. CONTRAST SENSITIVITY TEST - Gray Scale
        self.contrast_tests = {
            'name': 'Contrast Sensitivity Test',
            'type': 'contrast',
            'levels': [
                {'value': '█████', 'level': 100, 'desc': '100% Contrast - Black/White'},
                {'value': '▓▓▓▓▓', 'level': 80, 'desc': '80% Contrast'},
                {'value': '▒▒▒▒▒', 'level': 60, 'desc': '60% Contrast'},
                {'value': '░░░░░', 'level': 50, 'desc': '50% Contrast'},
                {'value': '·········', 'level': 40, 'desc': '40% Contrast'},
                {'value': '.........', 'level': 30, 'desc': '30% Contrast'},
                {'value': '         ', 'level': 20, 'desc': '20% Contrast - Very Light'}
            ],
            'current_index': 0
        }
        
        # 5. COLOR VISION TEST - Colored Text
        self.color_vision_tests = {
            'name': 'Color Vision Test',
            'type': 'color',
            'levels': [
                {'value': '12', 'color': 'red', 'bg': 'green', 'desc': 'Red 12 on Green'},
                {'value': '8', 'color': 'green', 'bg': 'red', 'desc': 'Green 8 on Red'},
                {'value': '6', 'color': 'blue', 'bg': 'yellow', 'desc': 'Blue 6 on Yellow'},
                {'value': '29', 'color': 'orange', 'bg': 'blue', 'desc': 'Orange 29 on Blue'},
                {'value': '5', 'color': 'purple', 'bg': 'green', 'desc': 'Purple 5 on Green'},
                {'value': '3', 'color': 'yellow', 'bg': 'blue', 'desc': 'Yellow 3 on Blue'},
                {'value': '15', 'color': 'red', 'bg': 'gray', 'desc': 'Red 15 on Gray'},
                {'value': '74', 'color': 'green', 'bg': 'brown', 'desc': 'Green 74 on Brown'}
            ],
            'current_index': 0
        }
        
        # 6. VISUAL FIELD TEST - Peripheral Vision
        self.field_tests = {
            'name': 'Visual Field Test',
            'type': 'field',
            'levels': [
                {'value': '⬤ CENTER', 'position': 'center', 'desc': 'Look at Center Dot'},
                {'value': '⬤ LEFT', 'position': 'left', 'desc': 'Look Left'},
                {'value': '⬤ RIGHT', 'position': 'right', 'desc': 'Look Right'},
                {'value': '⬤ UP', 'position': 'up', 'desc': 'Look Up'},
                {'value': '⬤ DOWN', 'position': 'down', 'desc': 'Look Down'},
                {'value': '⬤ UP-LEFT', 'position': 'upleft', 'desc': 'Look Up-Left'},
                {'value': '⬤ UP-RIGHT', 'position': 'upright', 'desc': 'Look Up-Right'},
                {'value': '⬤ DOWN-LEFT', 'position': 'downleft', 'desc': 'Look Down-Left'},
                {'value': '⬤ DOWN-RIGHT', 'position': 'downright', 'desc': 'Look Down-Right'}
            ],
            'current_index': 0
        }
        
        # 7. PHORIA TEST (Eye Alignment) - Cover Test
        self.phoria_tests = {
            'name': 'Eye Alignment Test',
            'type': 'phoria',
            'levels': [
                {'value': '⊙ FIXATION', 'target': 'center', 'desc': 'Fix on Center Dot'},
                {'value': '◉ COVER LEFT', 'target': 'cover_left', 'desc': 'Cover Left Eye'},
                {'value': '◉ COVER RIGHT', 'target': 'cover_right', 'desc': 'Cover Right Eye'},
                {'value': '◉ ALTERNATE', 'target': 'alternate', 'desc': 'Alternate Cover Test'},
                {'value': '◉ BOTH EYES', 'target': 'both', 'desc': 'Both Eyes Open'}
            ],
            'current_index': 0
        }
        
        # 8. ACCOMMODATION TEST (Focusing) - Near/Far
        self.accommodation_tests = {
            'name': 'Focusing Ability Test',
            'type': 'accommodation',
            'levels': [
                {'value': '⬤ FAR', 'distance': 'far', 'desc': 'Focus on Far Object'},
                {'value': '⬤ INTERMEDIATE', 'distance': 'intermediate', 'desc': 'Focus at Arm Length'},
                {'value': '⬤ NEAR', 'distance': 'near', 'desc': 'Focus on Near Object'},
                {'value': '⬤ VERY NEAR', 'distance': 'very_near', 'desc': 'Focus Very Close'},
                {'value': '⬤ RAPID CHANGE', 'distance': 'rapid', 'desc': 'Rapid Focus Change'}
            ],
            'current_index': 0
        }
        
        # Test sequence - All tests in order
        self.all_tests = [
            ('myopia', self.myopia_tests),
            ('hyperopia', self.hyperopia_tests),
            ('astigmatism', self.astigmatism_tests),
            ('contrast', self.contrast_tests),
            ('color', self.color_vision_tests),
            ('field', self.field_tests),
            ('phoria', self.phoria_tests),
            ('accommodation', self.accommodation_tests)
        ]
        
        self.current_test_index = 0
        self.current_test_type, self.current_test = self.all_tests[0]
        
        # Data storage
        self.eye_tracking_data = []
        self.test_results = []
        self.patient_name = "Patient"
        
        # Eye tracking
        self.left_eye_trajectory = []
        self.right_eye_trajectory = []
        self.timestamps = []
        self.start_time = time.time()
        
        print("✅ Hospital Grade Vision Screening System Ready!")
        print(f"📋 Total Tests: {len(self.all_tests)}")

    def detect_eyes(self, frame):
        """Detect eyes and return metrics"""
        if frame is None:
            return frame, {'detected': False}
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(100, 100)
        )
        
        eye_metrics = {
            'detected': False,
            'left_eye': None,
            'right_eye': None,
            'face_detected': len(faces) > 0,
            'timestamp': time.time()
        }
        
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            roi_gray = gray[y:y+h//2, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            if len(eyes) >= 2:
                eye_metrics['detected'] = True
                eyes = sorted(eyes, key=lambda e: e[0])
                
                for i, (ex, ey, ew, eh) in enumerate(eyes[:2]):
                    eye_x = x + ex + ew // 2
                    eye_y = y + ey + eh // 2
                    
                    if i == 0:
                        eye_metrics['left_eye'] = (eye_x, eye_y)
                    else:
                        eye_metrics['right_eye'] = (eye_x, eye_y)
                    
                    cv2.rectangle(frame, 
                                (x + ex, y + ey), 
                                (x + ex + ew, y + ey + eh), 
                                (0, 0, 255), 2)
                    cv2.circle(frame, (eye_x, eye_y), 3, (255, 255, 0), -1)
                
                # Store eye tracking
                self.timestamps.append(elapsed_time)
                self.left_eye_trajectory.append(eye_metrics['left_eye'][1] if eye_metrics['left_eye'] else 0)
                self.right_eye_trajectory.append(eye_metrics['right_eye'][1] if eye_metrics['right_eye'] else 0)
                
                self.eye_tracking_data.append({
                    'timestamp': elapsed_time,
                    'left_eye_y': eye_metrics['left_eye'][1] if eye_metrics['left_eye'] else 0,
                    'right_eye_y': eye_metrics['right_eye'][1] if eye_metrics['right_eye'] else 0
                })
        
        return frame, eye_metrics

    def create_test_pattern(self, test_type, level_data, width=400, height=300):
        """Create actual test pattern image"""
        pattern = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
        
        if test_type == 'myopia':
            # Show letter
            letter = level_data['value']
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = level_data['size'] / 50
            thickness = max(2, int(level_data['size'] / 20))
            text_size = cv2.getTextSize(letter, font, font_scale, thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (height + text_size[1]) // 2
            cv2.putText(pattern, letter, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
            
        elif test_type == 'hyperopia':
            # Show reading text
            text = level_data['value']
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = level_data['size'] / 50
            thickness = 1
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (height + text_size[1]) // 2
            cv2.putText(pattern, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
            
        elif test_type == 'astigmatism':
            # Draw lines based on angle
            center = (width//2, height//2)
            if level_data['angle'] == 'grid':
                # Draw grid
                for i in range(0, width, 20):
                    cv2.line(pattern, (i, 0), (i, height), (0, 0, 0), 1)
                for i in range(0, height, 20):
                    cv2.line(pattern, (0, i), (width, i), (0, 0, 0), 1)
            elif level_data['angle'] == 'dots':
                # Draw dot pattern
                for x in range(50, width, 50):
                    for y in range(50, height, 50):
                        cv2.circle(pattern, (x, y), 5, (0, 0, 0), -1)
            else:
                # Draw single line
                angle_rad = np.radians(level_data['angle'])
                length = min(width, height) // 2
                dx = int(length * np.cos(angle_rad))
                dy = int(length * np.sin(angle_rad))
                cv2.line(pattern, (center[0]-dx, center[1]-dy), 
                        (center[0]+dx, center[1]+dy), (0, 0, 0), 3)
                # Draw perpendicular line
                cv2.line(pattern, (center[0]-dy, center[1]+dx), 
                        (center[0]+dy, center[1]-dx), (0, 0, 0), 1)
            
        elif test_type == 'contrast':
            # Create contrast bars
            bar_height = height // 3
            level = level_data['level']
            gray_val = int(255 * level / 100)
            for i in range(3):
                y_start = i * bar_height
                color = (gray_val, gray_val, gray_val) if i % 2 == 0 else (255, 255, 255)
                cv2.rectangle(pattern, (0, y_start), (width, y_start + bar_height), color, -1)
            
        elif test_type == 'color':
            # Colored text
            color_map = {
                'red': (0, 0, 255),
                'green': (0, 255, 0),
                'blue': (255, 0, 0),
                'yellow': (0, 255, 255),
                'orange': (0, 165, 255),
                'purple': (128, 0, 128),
                'gray': (128, 128, 128),
                'brown': (42, 42, 165)
            }
            bg_map = {
                'red': (0, 0, 100),
                'green': (0, 100, 0),
                'blue': (100, 0, 0),
                'yellow': (100, 100, 0),
                'green': (0, 100, 0),
                'gray': (100, 100, 100),
                'brown': (42, 42, 42)
            }
            text_color = color_map.get(level_data['color'], (0, 0, 0))
            bg_color = bg_map.get(level_data['bg'], (255, 255, 255))
            pattern[:] = bg_color
            text = level_data['value']
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 2, 3)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (height + text_size[1]) // 2
            cv2.putText(pattern, text, (text_x, text_y), font, 2, text_color, 3)
            
        elif test_type == 'field':
            # Position indicator
            cv2.circle(pattern, (width//2, height//2), 10, (0, 0, 0), -1)
            pos_map = {
                'left': (width//4, height//2),
                'right': (3*width//4, height//2),
                'up': (width//2, height//4),
                'down': (width//2, 3*height//4),
                'upleft': (width//4, height//4),
                'upright': (3*width//4, height//4),
                'downleft': (width//4, 3*height//4),
                'downright': (3*width//4, 3*height//4)
            }
            if level_data['position'] in pos_map:
                x, y = pos_map[level_data['position']]
                cv2.circle(pattern, (x, y), 15, (0, 0, 0), -1)
            cv2.circle(pattern, (width//2, height//2), 5, (255, 0, 0), -1)
            
        elif test_type in ['phoria', 'accommodation']:
            # Simple fixation target
            cv2.circle(pattern, (width//2, height//2), 15, (0, 0, 0), 2)
            cv2.circle(pattern, (width//2, height//2), 5, (0, 0, 0), -1)
            cv2.line(pattern, (width//2-30, height//2), (width//2+30, height//2), (0, 0, 0), 2)
            cv2.line(pattern, (width//2, height//2-30), (width//2, height//2+30), (0, 0, 0), 2)
            
        return pattern

    def get_current_pattern(self):
        """Get current test pattern"""
        level = self.current_test['levels'][self.current_test['current_index']]
        return level

    def process_response(self, answer):
        """Process user response"""
        current_level = self.current_test['current_index']
        level_data = self.current_test['levels'][current_level]
        
        # Record result
        result = {
            'test': self.current_test['name'],
            'test_type': self.current_test_type,
            'level': current_level + 1,
            'level_data': level_data,
            'answer': answer,
            'timestamp': time.time()
        }
        self.test_results.append(result)
        
        print(f"✅ {self.current_test['name']} - Level {current_level + 1}: {answer}")
        
        # Move to next level
        self.current_test['current_index'] += 1
        
        # Check if test complete
        if self.current_test['current_index'] >= len(self.current_test['levels']):
            self.current_test['current_index'] = 0
            self.current_test_index += 1
            
            if self.current_test_index < len(self.all_tests):
                self.current_test_type, self.current_test = self.all_tests[self.current_test_index]
                print(f"➡️ Starting: {self.current_test['name']}")
        
        # Get next pattern
        next_level = self.get_current_pattern()
        
        # Create pattern image
        pattern_img = self.create_test_pattern(self.current_test_type, next_level)
        _, buffer = cv2.imencode('.jpg', pattern_img)
        pattern_base64 = base64.b64encode(buffer).decode()
        
        return {
            'success': True,
            'current_level': self.current_test['current_index'],
            'total_levels': len(self.current_test['levels']),
            'test_name': self.current_test['name'],
            'pattern_display': next_level['value'],
            'pattern_info': next_level.get('desc', ''),
            'test_type': self.current_test_type,
            'pattern_image': pattern_base64
        }

    def analyze_comprehensive_results(self):
        """Analyze all test results comprehensively"""
        if not self.test_results:
            return {
                'myopia': {'status': 'Not Tested', 'score': 0},
                'hyperopia': {'status': 'Not Tested', 'score': 0},
                'astigmatism': {'status': 'Not Tested', 'score': 0},
                'contrast': {'status': 'Not Tested', 'score': 0},
                'color': {'status': 'Not Tested', 'score': 0},
                'field': {'status': 'Not Tested', 'score': 0},
                'phoria': {'status': 'Not Tested', 'score': 0},
                'accommodation': {'status': 'Not Tested', 'score': 0},
                'overall_score': 0,
                'diagnosis': 'Incomplete Test',
                'recommendation': 'Please complete all vision tests'
            }
        
        # Analyze each test type
        results = {}
        total_score = 0
        test_count = 0
        
        for test_type, test_data in self.all_tests:
            test_results = [r for r in self.test_results if r['test_type'] == test_type]
            if test_results:
                correct = len([r for r in test_results if r['answer'] == 'yes'])
                total = len(test_results)
                score = (correct / total) * 100 if total > 0 else 0
                
                # Determine status based on score
                if score >= 80:
                    status = "Normal"
                elif score >= 60:
                    status = "Mild Issue"
                elif score >= 40:
                    status = "Moderate Issue"
                else:
                    status = "Severe Issue"
                
                results[test_type] = {
                    'score': score,
                    'status': status,
                    'correct': correct,
                    'total': total
                }
                total_score += score
                test_count += 1
        
        overall_score = total_score / test_count if test_count > 0 else 0
        
        # Generate comprehensive diagnosis
        diagnosis = self.generate_diagnosis(results, overall_score)
        recommendation = self.generate_recommendation(results, overall_score)
        
        return {
            **results,
            'overall_score': overall_score,
            'diagnosis': diagnosis,
            'recommendation': recommendation,
            'total_tests': test_count,
            'total_responses': len(self.test_results)
        }

    def generate_diagnosis(self, results, overall_score):
        """Generate medical diagnosis"""
        issues = []
        
        if 'myopia' in results and results['myopia']['score'] < 70:
            issues.append("Myopia (Near-sightedness)")
        if 'hyperopia' in results and results['hyperopia']['score'] < 70:
            issues.append("Hyperopia (Far-sightedness)")
        if 'astigmatism' in results and results['astigmatism']['score'] < 70:
            issues.append("Astigmatism")
        if 'contrast' in results and results['contrast']['score'] < 60:
            issues.append("Contrast Sensitivity Deficiency")
        if 'color' in results and results['color']['score'] < 60:
            issues.append("Color Vision Deficiency")
        if 'field' in results and results['field']['score'] < 70:
            issues.append("Visual Field Defect")
        if 'phoria' in results and results['phoria']['score'] < 70:
            issues.append("Eye Alignment Issue")
        if 'accommodation' in results and results['accommodation']['score'] < 70:
            issues.append("Focusing Difficulty")
        
        if not issues:
            return "Normal Vision - No significant issues detected"
        elif len(issues) == 1:
            return f"Diagnosis: {issues[0]}"
        else:
            return f"Diagnosis: Multiple conditions - {', '.join(issues)}"

    def generate_recommendation(self, results, overall_score):
        """Generate medical recommendation"""
        if overall_score >= 85:
            return "✅ Vision is excellent. Regular checkup in 1-2 years recommended."
        elif overall_score >= 70:
            return "✅ Vision is good. Annual checkup recommended."
        elif overall_score >= 60:
            return "⚠️ Mild vision issues detected. Consider glasses for specific activities."
        elif overall_score >= 50:
            return "⚠️ Moderate vision issues. Prescription glasses recommended for daily use."
        elif overall_score >= 40:
            return "❌ Significant vision issues. Strongly recommend comprehensive eye exam."
        else:
            return "🚨 URGENT: Severe vision issues detected. Immediate consultation with eye specialist required."

    def generate_eye_tracking_graph(self):
        """Generate eye tracking graph"""
        plt.figure(figsize=(14, 5))
        
        # Left Eye
        plt.subplot(1, 2, 1)
        if len(self.timestamps) > 5:
            n = min(100, len(self.timestamps))
            plt.plot(self.timestamps[-n:], self.left_eye_trajectory[-n:], 
                    'r-', linewidth=2, label='Left Eye')
        else:
            x = np.linspace(0, 10, 50)
            plt.plot(x, 5 + 2*np.sin(x), 'r-', linewidth=2, label='Left Eye')
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Eye Position (pixels)')
        plt.title('Left Eye Movement', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Right Eye
        plt.subplot(1, 2, 2)
        if len(self.timestamps) > 5:
            n = min(100, len(self.timestamps))
            plt.plot(self.timestamps[-n:], self.right_eye_trajectory[-n:], 
                    'b-', linewidth=2, label='Right Eye')
        else:
            x = np.linspace(0, 10, 50)
            plt.plot(x, 5 + 2*np.cos(x), 'b-', linewidth=2, label='Right Eye')
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Eye Position (pixels)')
        plt.title('Right Eye Movement', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.suptitle('Eye Tracking Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        graph_data = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return graph_data

    def generate_test_results_graph(self, results):
        """Generate test results bar chart"""
        plt.figure(figsize=(12, 6))
        
        tests = []
        scores = []
        colors = []
        
        test_names = {
            'myopia': 'Myopia',
            'hyperopia': 'Hyperopia',
            'astigmatism': 'Astigmatism',
            'contrast': 'Contrast',
            'color': 'Color Vision',
            'field': 'Visual Field',
            'phoria': 'Eye Alignment',
            'accommodation': 'Focus'
        }
        
        for test_type, test_name in test_names.items():
            if test_type in results:
                tests.append(test_name)
                score = results[test_type]['score']
                scores.append(score)
                
                if score >= 80:
                    colors.append('#2ecc71')  # Green
                elif score >= 60:
                    colors.append('#f39c12')  # Orange
                elif score >= 40:
                    colors.append('#e74c3c')  # Red
                else:
                    colors.append('#c0392b')  # Dark Red
        
        bars = plt.bar(tests, scores, color=colors, edgecolor='black', linewidth=2)
        plt.ylim(0, 100)
        plt.ylabel('Score (%)', fontsize=12)
        plt.title('Vision Test Results', fontweight='bold', fontsize=16)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{score:.0f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        graph_data = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return graph_data

    def generate_diagnosis_graph(self, results):
        """Generate diagnosis pie chart"""
        plt.figure(figsize=(8, 8))
        
        # Count normal vs abnormal
        normal_count = 0
        mild_count = 0
        moderate_count = 0
        severe_count = 0
        
        for test_type in ['myopia', 'hyperopia', 'astigmatism', 'contrast', 
                         'color', 'field', 'phoria', 'accommodation']:
            if test_type in results:
                score = results[test_type]['score']
                if score >= 80:
                    normal_count += 1
                elif score >= 60:
                    mild_count += 1
                elif score >= 40:
                    moderate_count += 1
                else:
                    severe_count += 1
        
        sizes = []
        labels = []
        colors = []
        
        if normal_count > 0:
            sizes.append(normal_count)
            labels.append(f'Normal ({normal_count})')
            colors.append('#2ecc71')
        if mild_count > 0:
            sizes.append(mild_count)
            labels.append(f'Mild ({mild_count})')
            colors.append('#f39c12')
        if moderate_count > 0:
            sizes.append(moderate_count)
            labels.append(f'Moderate ({moderate_count})')
            colors.append('#e74c3c')
        if severe_count > 0:
            sizes.append(severe_count)
            labels.append(f'Severe ({severe_count})')
            colors.append('#c0392b')
        
        if sizes:
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                   shadow=True, startangle=90, textprops={'fontweight': 'bold'})
            plt.axis('equal')
            plt.title('Vision Health Overview', fontweight='bold', fontsize=16)
        
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        graph_data = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return graph_data

    def save_results(self):
        """Save results to CSV and TXT"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = self.analyze_comprehensive_results()
        
        # CSV File
        csv_file = os.path.join(self.results_dir, f"vision_test_results_{timestamp}.csv")
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['VISION SCREENING REPORT'])
            writer.writerow(['Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            writer.writerow(['Patient:', self.patient_name])
            writer.writerow([])
            writer.writerow(['OVERALL RESULTS'])
            writer.writerow(['Overall Score', f"{results['overall_score']:.1f}%"])
            writer.writerow(['Diagnosis', results['diagnosis']])
            writer.writerow(['Recommendation', results['recommendation']])
            writer.writerow([])
            writer.writerow(['TEST DETAILS'])
            writer.writerow(['Test', 'Score (%)', 'Status', 'Correct/Total'])
            
            test_names = {
                'myopia': 'Myopia (Distance)',
                'hyperopia': 'Hyperopia (Reading)',
                'astigmatism': 'Astigmatism',
                'contrast': 'Contrast Sensitivity',
                'color': 'Color Vision',
                'field': 'Visual Field',
                'phoria': 'Eye Alignment',
                'accommodation': 'Focusing Ability'
            }
            
            for test_key, test_name in test_names.items():
                if test_key in results:
                    writer.writerow([
                        test_name,
                        f"{results[test_key]['score']:.1f}",
                        results[test_key]['status'],
                        f"{results[test_key]['correct']}/{results[test_key]['total']}"
                    ])
        
        # TXT Report
        txt_file = os.path.join(self.results_dir, f"vision_report_{timestamp}.txt")
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("        COMPREHENSIVE VISION SCREENING REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Patient: {self.patient_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Time: {datetime.now().strftime('%H:%M:%S')}\n\n")
            f.write("-" * 70 + "\n")
            f.write("OVERALL ASSESSMENT:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Overall Score: {results['overall_score']:.1f}%\n")
            f.write(f"Diagnosis: {results['diagnosis']}\n")
            f.write(f"Recommendation: {results['recommendation']}\n\n")
            f.write("-" * 70 + "\n")
            f.write("DETAILED TEST RESULTS:\n")
            f.write("-" * 70 + "\n")
            
            for test_key, test_name in test_names.items():
                if test_key in results:
                    f.write(f"\n{test_name}:\n")
                    f.write(f"  Score: {results[test_key]['score']:.1f}%\n")
                    f.write(f"  Status: {results[test_key]['status']}\n")
                    f.write(f"  Correct: {results[test_key]['correct']}/{results[test_key]['total']}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("⚠️ This is a screening tool. Please consult an eye care professional.\n")
            f.write("=" * 70 + "\n")
        
        print(f"✅ CSV saved: {csv_file}")
        print(f"✅ TXT saved: {txt_file}")
        
        return csv_file, txt_file


# ==========================================
# FLASK WEB SERVER
# ==========================================
screener = VisionScreeningSystem()
camera = None
camera_lock = threading.Lock()
is_running = False
camera_initialized = False

def init_camera():
    global camera, camera_initialized
    with camera_lock:
        if camera_initialized and camera is not None:
            return True
        
        if camera is not None:
            try:
                camera.release()
            except:
                pass
            camera = None
        
        for i in range(3):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        camera = cap
                        camera_initialized = True
                        print(f"✅ Camera {i} initialized")
                        return True
                    else:
                        cap.release()
            except:
                continue
        
        print("❌ No camera found")
        return False

def release_camera():
    global camera, camera_initialized
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
            camera_initialized = False

def generate_frames():
    global is_running, camera_initialized
    is_running = True
    
    # No camera frame
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank, "NO CAMERA FOUND", (180, 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    ret, buffer = cv2.imencode('.jpg', blank)
    no_camera = buffer.tobytes()
    
    camera_ok = init_camera()
    last_frame = time.time()
    
    while is_running:
        if not camera_ok or camera is None:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + no_camera + b'\r\n')
            time.sleep(0.5)
            if int(time.time()) % 5 == 0:
                camera_ok = init_camera()
            continue
        
        with camera_lock:
            if camera is None:
                continue
            success, frame = camera.read()
        
        if not success:
            camera_ok = False
            continue
        
        # Control frame rate
        if time.time() - last_frame < 0.033:
            time.sleep(0.001)
            continue
        last_frame = time.time()
        
        try:
            processed, metrics = screener.detect_eyes(frame)
            
            # Add test info
            level = screener.get_current_pattern()
            cv2.putText(processed, f"Test: {screener.current_test['name']}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(processed, f"Pattern: {level['value']}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(processed, f"{level.get('desc', '')}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            status = "✓ Eyes Detected" if metrics['detected'] else "✗ No Eyes"
            color = (0, 255, 0) if metrics['detected'] else (0, 0, 255)
            cv2.putText(processed, status, (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Show progress
            progress = f"Level {screener.current_test['current_index']}/{len(screener.current_test['levels'])}"
            cv2.putText(processed, progress, (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', processed)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            print(f"Frame error: {e}")
            continue


# ==========================================
# FLASK ROUTES
# ==========================================
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/health')
def health():
    return jsonify({
        'camera': 'connected' if camera_initialized and camera else 'disconnected',
        'test': screener.current_test['name'],
        'responses': len(screener.test_results)
    })

@app.route('/get_test_info')
def get_test_info():
    level = screener.get_current_pattern()
    # Create pattern image
    pattern_img = screener.create_test_pattern(screener.current_test_type, level)
    _, buffer = cv2.imencode('.jpg', pattern_img)
    pattern_base64 = base64.b64encode(buffer).decode()
    
    return jsonify({
        'test_name': screener.current_test['name'],
        'current_level': screener.current_test['current_index'],
        'total_levels': len(screener.current_test['levels']),
        'pattern_display': level['value'],
        'pattern_info': level.get('desc', ''),
        'pattern_image': pattern_base64,
        'responses': len(screener.test_results)
    })

@app.route('/process_response', methods=['POST'])
def process_response():
    try:
        answer = request.json.get('response')
        result = screener.process_response(answer)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        results = screener.analyze_comprehensive_results()
        eye_graph = screener.generate_eye_tracking_graph()
        test_graph = screener.generate_test_results_graph(results)
        diagnosis_graph = screener.generate_diagnosis_graph(results)
        csv_file, txt_file = screener.save_results()
        
        # Read report content
        with open(txt_file, 'r', encoding='utf-8') as f:
            report = f.read()
        
        return jsonify({
            'success': True,
            'results': results,
            'graphs': {
                'eye_tracking': eye_graph,
                'test_results': test_graph,
                'diagnosis': diagnosis_graph
            },
            'report': report
        })
    except Exception as e:
        print(f"Report error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/reset_test')
def reset_test():
    screener.current_test_index = 0
    screener.current_test_type, screener.current_test = screener.all_tests[0]
    for _, test in screener.all_tests:
        test['current_index'] = 0
    screener.test_results = []
    screener.eye_tracking_data = []
    screener.left_eye_trajectory = []
    screener.right_eye_trajectory = []
    screener.timestamps = []
    screener.start_time = time.time()
    return jsonify({'success': True})

@app.route('/stop_feed')
def stop_feed():
    global is_running
    is_running = False
    release_camera()
    return jsonify({'status': 'stopped'})


# ==========================================
# HTML TEMPLATE - WITH ON-SCREEN RESULTS
# ==========================================
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hospital Grade Vision Screening System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #1a2a3a 0%, #0f1a24 100%);
            color: white;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1600px; margin: 0 auto; }
        h1 {
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 20px;
            color: #00d2ff;
            text-shadow: 0 0 20px rgba(0,210,255,0.5);
        }
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        
        /* Video Section */
        .video-section {
            background: rgba(0,0,0,0.5);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid #00d2ff;
        }
        .video-container {
            position: relative;
            width: 100%;
            background: #000;
            border-radius: 10px;
            overflow: hidden;
            border: 3px solid #00d2ff;
        }
        #videoFeed { width: 100%; height: auto; display: block; }
        .camera-status {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.8);
            padding: 5px 15px;
            border-radius: 20px;
            border-left: 4px solid #00ff00;
            font-size: 14px;
        }
        
        /* Test Pattern Display */
        .pattern-container {
            background: #000;
            border-radius: 10px;
            padding: 10px;
            margin: 15px 0;
            text-align: center;
            border: 2px solid #00d2ff;
            min-height: 200px;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        #patternImage {
            max-width: 100%;
            max-height: 250px;
            border-radius: 5px;
        }
        
        /* Test Section */
        .test-section {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid #ffd700;
        }
        .test-panel {
            background: rgba(0,0,0,0.5);
            border-radius: 10px;
            padding: 20px;
        }
        .test-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ffd700;
        }
        .test-title {
            font-size: 1.3em;
            font-weight: bold;
            color: #ffd700;
        }
        .progress {
            background: #333;
            height: 12px;
            border-radius: 6px;
            margin: 15px 0;
        }
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #00ff00, #ffff00);
            width: 0%;
            transition: width 0.3s;
            border-radius: 6px;
        }
        .controls {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin: 20px 0;
        }
        .btn {
            padding: 15px;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            color: white;
        }
        .btn-yes {
            background: linear-gradient(45deg, #00b09b, #96c93d);
        }
        .btn-no {
            background: linear-gradient(45deg, #ff416c, #ff4b2b);
        }
        .btn-report {
            background: linear-gradient(45deg, #667eea, #764ba2);
            width: 100%;
            margin-top: 10px;
        }
        .btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.3);
        }
        
        /* Results Section */
        .results-section {
            margin-top: 30px;
            background: rgba(0,0,0,0.9);
            border-radius: 15px;
            padding: 30px;
            border: 2px solid #00ff00;
            display: none;
        }
        .results-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .score-circle {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            background: linear-gradient(45deg, #00b09b, #96c93d);
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 36px;
            font-weight: bold;
            border: 3px solid white;
        }
        .diagnosis-box {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            border-left: 5px solid #ffd700;
        }
        .test-results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .test-result-card {
            background: rgba(0,0,0,0.5);
            border-radius: 8px;
            padding: 15px;
            border: 1px solid #00d2ff;
        }
        .test-result-card h4 {
            color: #ffd700;
            margin-bottom: 10px;
        }
        .test-score {
            font-size: 28px;
            font-weight: bold;
        }
        .test-status {
            font-size: 14px;
            margin-top: 5px;
            font-weight: bold;
        }
        .status-normal { color: #00ff00; }
        .status-mild { color: #ffff00; }
        .status-moderate { color: #ffa500; }
        .status-severe { color: #ff0000; }
        
        .graph-container {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin: 30px 0;
        }
        .graph-card {
            background: white;
            border-radius: 10px;
            padding: 15px;
        }
        .graph-card img {
            width: 100%;
            height: auto;
        }
        .graph-card h3 {
            color: #333;
            margin-bottom: 10px;
            text-align: center;
        }
        
        @media (max-width: 1200px) {
            .main-content { grid-template-columns: 1fr; }
            .graph-container { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏥 Hospital Grade Vision Screening System</h1>
        
        <div class="main-content">
            <!-- Video Section -->
            <div class="video-section">
                <div class="video-container">
                    <img id="videoFeed" src="/video_feed" alt="Video Feed">
                    <div class="camera-status" id="cameraStatus">📷 Initializing...</div>
                </div>
                <div style="margin-top: 15px; text-align: center;">
                    <span id="testProgress">Test: 1/8 | Level: 0/0</span>
                </div>
            </div>
            
            <!-- Test Section -->
            <div class="test-section">
                <div class="test-panel">
                    <div class="test-header">
                        <span class="test-title" id="testName">Myopia Test</span>
                        <span id="levelInfo">Level 1/13</span>
                    </div>
                    
                    <div class="progress">
                        <div class="progress-bar" id="progressBar"></div>
                    </div>
                    
                    <div class="pattern-container">
                        <img id="patternImage" src="" alt="Test Pattern">
                    </div>
                    
                    <div class="controls">
                        <button class="btn btn-yes" onclick="sendResponse('yes')">✅ YES - Can See</button>
                        <button class="btn btn-no" onclick="sendResponse('no')">❌ NO - Cannot See</button>
                    </div>
                    
                    <button class="btn btn-report" onclick="generateReport()">📊 Generate Complete Report</button>
                    <button class="btn btn-report" onclick="resetTest()" style="background: linear-gradient(45deg, #f12711, #f5af19); margin-top: 10px;">🔄 Reset All Tests</button>
                </div>
                
                <div id="responseMessage" style="text-align: center; margin-top: 15px; padding: 10px; border-radius: 5px;"></div>
            </div>
        </div>
        
        <!-- Results Section - ON SCREEN -->
        <div id="resultsSection" class="results-section">
            <div class="results-header">
                <h2>📋 Vision Screening Report</h2>
                <div class="score-circle" id="overallScore">0%</div>
            </div>
            
            <div class="diagnosis-box">
                <h3 id="diagnosisText" style="color: #ffd700;">Analyzing...</h3>
                <p id="recommendationText" style="margin-top: 10px; font-size: 16px;"></p>
            </div>
            
            <h3>📊 Test-by-Test Results</h3>
            <div class="test-results-grid" id="testResultsGrid"></div>
            
            <h3>📈 Visual Analysis</h3>
            <div class="graph-container" id="graphContainer"></div>
            
            <div style="text-align: center; margin-top: 20px;">
                <button onclick="printReport()" class="btn" style="background: linear-gradient(45deg, #667eea, #764ba2); padding: 15px 40px;">
                    🖨️ Print Report
                </button>
            </div>
        </div>
    </div>
    
    <script>
        // Update test info every second
        setInterval(updateTestInfo, 1000);
        setInterval(updateCameraStatus, 2000);
        
        function updateTestInfo() {
            fetch('/get_test_info')
                .then(res => res.json())
                .then(data => {
                    document.getElementById('testName').innerHTML = data.test_name;
                    document.getElementById('levelInfo').innerHTML = 
                        `Level ${data.current_level + 1}/${data.total_levels}`;
                    document.getElementById('patternImage').src = 'data:image/jpeg;base64,' + data.pattern_image;
                    document.getElementById('testProgress').innerHTML = 
                        `Test: ${data.test_name} | Level ${data.current_level + 1}/${data.total_levels}`;
                    
                    const progress = ((data.current_level) / data.total_levels) * 100;
                    document.getElementById('progressBar').style.width = progress + '%';
                })
                .catch(e => console.log('Update error:', e));
        }
        
        function updateCameraStatus() {
            fetch('/health')
                .then(res => res.json())
                .then(data => {
                    const status = document.getElementById('cameraStatus');
                    if (data.camera === 'connected') {
                        status.innerHTML = '📷 Camera Connected';
                        status.style.borderLeftColor = '#00ff00';
                    } else {
                        status.innerHTML = '📷 No Camera Found';
                        status.style.borderLeftColor = '#ff0000';
                    }
                })
                .catch(e => console.log('Status error:', e));
        }
        
        async function sendResponse(answer) {
            try {
                const res = await fetch('/process_response', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({response: answer})
                });
                const data = await res.json();
                
                if (data.success) {
                    document.getElementById('testName').innerHTML = data.test_name;
                    document.getElementById('levelInfo').innerHTML = 
                        `Level ${data.current_level + 1}/${data.total_levels}`;
                    document.getElementById('patternImage').src = 'data:image/jpeg;base64,' + data.pattern_image;
                    
                    const progress = (data.current_level / data.total_levels) * 100;
                    document.getElementById('progressBar').style.width = progress + '%';
                    
                    const msg = document.getElementById('responseMessage');
                    msg.innerHTML = answer === 'yes' ? '✅ Clear vision recorded' : '❌ Blurry vision recorded';
                    msg.style.background = answer === 'yes' ? 'rgba(0,255,0,0.2)' : 'rgba(255,0,0,0.2)';
                    msg.style.color = answer === 'yes' ? '#00ff00' : '#ff6666';
                    setTimeout(() => msg.innerHTML = '', 2000);
                }
            } catch (e) {
                console.error('Response error:', e);
                alert('Error sending response');
            }
        }
        
        async function generateReport() {
            const btn = event.target;
            const originalText = btn.innerHTML;
            btn.innerHTML = '⏳ Generating Report...';
            btn.disabled = true;
            
            try {
                const res = await fetch('/generate_report', {method: 'POST'});
                const data = await res.json();
                
                if (data.success) {
                    // Show results section
                    document.getElementById('resultsSection').style.display = 'block';
                    
                    // Update overall score
                    document.getElementById('overallScore').innerHTML = 
                        `${data.results.overall_score.toFixed(1)}%`;
                    
                    // Update diagnosis
                    document.getElementById('diagnosisText').innerHTML = data.results.diagnosis;
                    document.getElementById('recommendationText').innerHTML = 
                        `💡 ${data.results.recommendation}`;
                    
                    // Build test results grid
                    const testNames = {
                        'myopia': 'Myopia (Distance)',
                        'hyperopia': 'Hyperopia (Reading)',
                        'astigmatism': 'Astigmatism',
                        'contrast': 'Contrast',
                        'color': 'Color Vision',
                        'field': 'Visual Field',
                        'phoria': 'Eye Alignment',
                        'accommodation': 'Focusing'
                    };
                    
                    let gridHtml = '';
                    for (const [key, name] of Object.entries(testNames)) {
                        if (data.results[key]) {
                            const test = data.results[key];
                            const statusClass = test.status.toLowerCase().replace(' ', '-');
                            gridHtml += `
                                <div class="test-result-card">
                                    <h4>${name}</h4>
                                    <div class="test-score">${test.score.toFixed(1)}%</div>
                                    <div class="test-status status-${statusClass}">${test.status}</div>
                                    <div style="font-size: 12px; margin-top: 5px;">${test.correct}/${test.total} correct</div>
                                </div>
                            `;
                        }
                    }
                    document.getElementById('testResultsGrid').innerHTML = gridHtml;
                    
                    // Show graphs
                    document.getElementById('graphContainer').innerHTML = `
                        <div class="graph-card">
                            <h3>👁️ Eye Tracking</h3>
                            <img src="data:image/png;base64,${data.graphs.eye_tracking}">
                        </div>
                        <div class="graph-card">
                            <h3>📊 Test Results</h3>
                            <img src="data:image/png;base64,${data.graphs.test_results}">
                        </div>
                        <div class="graph-card">
                            <h3>📈 Vision Health</h3>
                            <img src="data:image/png;base64,${data.graphs.diagnosis}">
                        </div>
                    `;
                    
                    // Scroll to results
                    document.getElementById('resultsSection').scrollIntoView({behavior: 'smooth'});
                }
            } catch (e) {
                console.error('Report error:', e);
                alert('Error generating report. Please try again.');
            }
            
            btn.innerHTML = originalText;
            btn.disabled = false;
        }
        
        function printReport() {
            window.print();
        }
        
        async function resetTest() {
            if (confirm('Reset all test progress?')) {
                await fetch('/reset_test');
                updateTestInfo();
                document.getElementById('resultsSection').style.display = 'none';
            }
        }
        
        window.addEventListener('beforeunload', () => fetch('/stop_feed'));
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    print("=" * 70)
    print("🏥 HOSPITAL GRADE VISION SCREENING SYSTEM")
    print("=" * 70)
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f"📍 Local URL: http://localhost:5000")
        print(f"📍 Network URL: http://{local_ip}:5000")
    except:
        print(f"📍 Local URL: http://localhost:5000")
    print("=" * 70)
    print("✅ 8 Complete Vision Tests with Proper Shapes/Text:")
    print("   • Myopia Test - Letters (E, H, N, O, S...)")
    print("   • Hyperopia Test - Reading Text")
    print("   • Astigmatism Test - Lines (─, │, ╱, ╲, Grid, Dots)")
    print("   • Contrast Test - Gray Scale (█, ▓, ▒, ░, .)")
    print("   • Color Vision Test - Colored Numbers")
    print("   • Visual Field Test - Position Dots")
    print("   • Eye Alignment Test - Fixation Targets")
    print("   • Focusing Test - Near/Far Targets")
    print("=" * 70)
    print("Press CTRL+C to stop")
    print("=" * 70)
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        release_camera()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        release_camera()