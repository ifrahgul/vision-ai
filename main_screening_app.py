import cv2
import numpy as np
import time
import sys
import os
from vision_screening_system import VisionScreeningSystem

def display_welcome_screen(width, height):
    """Display welcome screen with instructions"""
    welcome_screen = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Title
    cv2.putText(welcome_screen, "👁️ AI VISION SCREENING SYSTEM", 
                (width//2 - 300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    
    # Instructions
    instructions = [
        "INSTRUCTIONS:",
        "1. Sit 2-3 feet from the camera",
        "2. Ensure good lighting on your face",
        "3. Keep eyes on the test patterns",
        "4. Use keyboard to respond:",
        "",
        "🎮 CONTROLS:",
        "  'Y' - Yes, I can see clearly",
        "  'N' - No, it's blurry",
        "  'S' - Skip to next test",
        "  'R' - Repeat current pattern",
        "  'Q' - Quit and generate report",
        "",
        "Press any key to start..."
    ]
    
    y_pos = 150
    for i, line in enumerate(instructions):
        color = (0, 255, 255) if i == 0 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = 0.7 if i > 0 else 0.8
        
        cv2.putText(welcome_screen, line, 
                   (width//2 - 300, y_pos + i*40), 
                   font, size, color, 1 if i > 0 else 2)
    
    cv2.imshow("AI Vision Screening", welcome_screen)
    cv2.waitKey(0)

def get_patient_info(width, height):
    """Get basic patient information"""
    info_screen = np.zeros((height, width, 3), dtype=np.uint8)
    
    cv2.putText(info_screen, "PATIENT INFORMATION", 
                (width//2 - 200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    
    cv2.putText(info_screen, "Press '1' for Adult (18+ years)", 
                (width//2 - 200, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.putText(info_screen, "Press '2' for Child (Under 18)", 
                (width//2 - 200, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.putText(info_screen, "Press '3' for Senior (60+ years)", 
                (width//2 - 200, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imshow("AI Vision Screening", info_screen)
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('1'):
            return {'age_group': 'Adult', 'name': 'Adult User'}
        elif key == ord('2'):
            return {'age_group': 'Child', 'name': 'Child User'}
        elif key == ord('3'):
            return {'age_group': 'Senior', 'name': 'Senior User'}
        elif key == 27:  # ESC
            return None

def create_test_pattern_frame(test_type, index, pattern_width, pattern_height):
    """Create a separate frame for test pattern"""
    pattern_frame = np.zeros((pattern_height, pattern_width, 3), dtype=np.uint8)
    
    # Add test pattern to this frame
    if test_type == 'snellen_chart':
        letters = ['E', 'H', 'N', 'O', 'S', 'V', 'Z', 'T', 'L', 'C', 'F', 'P', 'D']
        sizes = [100, 80, 60, 50, 40, 30, 25, 20, 15, 12, 10, 8, 6]
        
        if index < len(letters):
            letter = letters[index]
            size = sizes[index]
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = size / 50
            thickness = max(2, int(size / 20))
            
            text_size = cv2.getTextSize(letter, font, font_scale, thickness)[0]
            text_x = (pattern_width - text_size[0]) // 2
            text_y = (pattern_height + text_size[1]) // 2
            
            cv2.putText(pattern_frame, letter, (text_x, text_y), 
                      font, font_scale, (255, 255, 255), thickness)
            
            # Add border
            cv2.rectangle(pattern_frame, (0, 0), (pattern_width-1, pattern_height-1), 
                         (0, 255, 255), 3)
            
            # Add instruction
            cv2.putText(pattern_frame, "Can you read this letter?", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    elif test_type == 'reading_test':
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Pack my box with five dozen liquor jugs.",
            "How vexingly quick daft zebras jump!",
            "Bright vixens jump; dozy fowl quack.",
            "Jinxed wizards pluck ivy from the big quilt."
        ]
        font_sizes = [40, 35, 30, 25, 20]
        
        if index < len(texts):
            text = texts[index]
            font_size = font_sizes[index]
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = font_size / 40
            thickness = 1
            
            # Split text
            words = text.split()
            lines = []
            current_line = ""
            
            for word in words:
                test_line = current_line + " " + word if current_line else word
                test_width = cv2.getTextSize(test_line, font, font_scale, thickness)[0][0]
                
                if test_width < pattern_width - 50:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word
            
            if current_line:
                lines.append(current_line)
            
            # Display lines
            line_height = int(font_size * 1.5)
            start_y = (pattern_height - len(lines) * line_height) // 2
            
            for i, line in enumerate(lines):
                text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
                text_x = (pattern_width - text_size[0]) // 2
                text_y = start_y + i * line_height
                cv2.putText(pattern_frame, line, (text_x, text_y), 
                          font, font_scale, (255, 255, 255), thickness)
            
            cv2.rectangle(pattern_frame, (0, 0), (pattern_width-1, pattern_height-1), 
                         (0, 255, 255), 3)
    
    elif test_type == 'radial_lines':
        angles = [0, 90, 45, 135, 0]
        line_widths = [5, 4, 3, 2, 1]
        
        if index < len(angles):
            center_x, center_y = pattern_width // 2, pattern_height // 2
            radius = min(pattern_width, pattern_height) // 3
            angle = angles[index]
            line_width = line_widths[index]
            
            end_x1 = int(center_x + radius * np.cos(np.radians(angle)))
            end_y1 = int(center_y + radius * np.sin(np.radians(angle)))
            end_x2 = int(center_x - radius * np.cos(np.radians(angle)))
            end_y2 = int(center_y - radius * np.sin(np.radians(angle)))
            
            cv2.line(pattern_frame, (center_x, center_y), (end_x1, end_y1), 
                    (255, 255, 255), line_width)
            cv2.line(pattern_frame, (center_x, center_y), (end_x2, end_y2), 
                    (255, 255, 255), line_width)
            
            cv2.rectangle(pattern_frame, (0, 0), (pattern_width-1, pattern_height-1), 
                         (0, 255, 255), 3)
            
            cv2.putText(pattern_frame, "Are lines equally sharp?", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    elif test_type == 'contrast_grid':
        levels = [100, 80, 60, 40, 20]
        
        if index < len(levels):
            contrast = levels[index]
            grid_size = 8
            cell_size = min(pattern_width, pattern_height) // (grid_size + 4)
            
            start_x = (pattern_width - grid_size * cell_size) // 2
            start_y = (pattern_height - grid_size * cell_size) // 2
            
            for i in range(grid_size):
                for j in range(grid_size):
                    x = start_x + i * cell_size
                    y = start_y + j * cell_size
                    
                    if (i + j) % 2 == 0:
                        intensity = 255 - contrast
                    else:
                        intensity = contrast
                    
                    cv2.rectangle(pattern_frame, (x, y), (x + cell_size, y + cell_size), 
                                (intensity, intensity, intensity), -1)
                    cv2.rectangle(pattern_frame, (x, y), (x + cell_size, y + cell_size), 
                                (100, 100, 100), 1)
            
            cv2.rectangle(pattern_frame, (0, 0), (pattern_width-1, pattern_height-1), 
                         (0, 255, 255), 3)
            
            cv2.putText(pattern_frame, f"Contrast: {contrast}%", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    return pattern_frame

def main():
    print("=" * 70)
    print("👁️ AI VISION SCREENING SYSTEM")
    print("=" * 70)
    print("Detect eyes → Show test patterns → Track responses → Generate report")
    print("=" * 70)
    
    # Initialize system
    try:
        screener = VisionScreeningSystem()
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam. Trying camera index 1...")
        cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("❌ Cannot open any webcam")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📷 Camera: {width}x{height}")
    
    # Display welcome screen
    display_welcome_screen(width, height)
    
    # Get patient info
    patient_info = get_patient_info(width, height)
    if patient_info is None:
        print("❌ No patient information provided")
        cap.release()
        cv2.destroyAllWindows()
        return
    
    print(f"\n👤 Patient: {patient_info['name']} ({patient_info['age_group']})")
    
    # Test sequence
    test_sequence = [
        ('distance', 'snellen_chart'),
        ('near', 'reading_test'),
        ('astigmatism', 'radial_lines'),
        ('contrast', 'contrast_grid')
    ]
    
    current_test_index = 0
    test_responses = []
    test_start_time = time.time()
    
    cv2.namedWindow("AI Vision Screening", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Vision Screening", 1000, 700)
    
    print("\n🔍 Starting vision screening...")
    print("Please respond to each test pattern.")
    
    # Define pattern window size
    pattern_width = 600
    pattern_height = 400
    
    try:
        while current_test_index < len(test_sequence):
            test_key, test_type = test_sequence[current_test_index]
            test_info = screener.tests[test_key]
            
            # Get current test index for this test type
            test_index = test_info['current_index']
            
            # Check if we've completed all patterns for this test
            max_patterns = len(
                test_info.get('letters') or 
                test_info.get('texts') or 
                test_info.get('patterns') or 
                test_info.get('levels')
            )
            
            if test_index >= max_patterns:
                test_info['current_index'] = 0
                current_test_index += 1
                if current_test_index >= len(test_sequence):
                    break
                continue
            
            # Reset for new pattern
            frame_count = 0
            pattern_description = ""
            
            print(f"\n▶️ Test: {test_info['name']}")
            print(f"   Pattern {test_index + 1} of {max_patterns}")
            
            # Create pattern frame once
            pattern_frame = create_test_pattern_frame(test_type, test_index, pattern_width, pattern_height)
            
            # Create pattern description
            if test_type == 'snellen_chart':
                pattern_description = f"Letter: {test_info['letters'][test_index]} (Size: {test_info['sizes'][test_index]}px)"
            elif test_type == 'reading_test':
                pattern_description = f"Reading text (Font: {test_info['font_sizes'][test_index]}px)"
            elif test_type == 'radial_lines':
                pattern_description = f"Astigmatism lines (Width: {test_info['line_widths'][test_index]})"
            elif test_type == 'contrast_grid':
                pattern_description = f"Contrast grid ({test_info['levels'][test_index]}%)"
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Camera error")
                    break
                
                # Process frame for eye detection
                processed_frame, eye_metrics = screener.detect_eyes(frame.copy())
                
                # Overlay pattern on camera feed
                x_offset = width - pattern_width - 20
                y_offset = (height - pattern_height) // 2
                
                # Create ROI and blend pattern
                roi = processed_frame[y_offset:y_offset+pattern_height, x_offset:x_offset+pattern_width]
                
                # Blend pattern with ROI (50% transparency)
                alpha = 0.7
                beta = 1.0 - alpha
                blended = cv2.addWeighted(roi, alpha, pattern_frame, beta, 0)
                processed_frame[y_offset:y_offset+pattern_height, x_offset:x_offset+pattern_width] = blended
                
                # Add border around pattern
                cv2.rectangle(processed_frame, 
                            (x_offset, y_offset), 
                            (x_offset+pattern_width, y_offset+pattern_height), 
                            (0, 255, 255), 3)
                
                # Display test info
                cv2.putText(processed_frame, test_info['name'], (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                cv2.putText(processed_frame, pattern_description, (10, 70),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.putText(processed_frame, f"Pattern {test_index + 1}/{max_patterns}", 
                          (width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 100), 2)
                
                # Display instructions
                cv2.putText(processed_frame, "Press 'Y' if clear, 'N' if blurry", (10, height - 100),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.putText(processed_frame, "'S'=Next test | 'R'=Repeat | 'Q'=Quit", (10, height - 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                # Progress bar
                progress = (test_index + 1) / max_patterns
                progress_width = int(width * 0.6 * progress)
                cv2.rectangle(processed_frame, (width//2 - 300, height - 30), 
                            (width//2 - 300 + progress_width, height - 10), 
                            (0, 200, 0), -1)
                cv2.rectangle(processed_frame, (width//2 - 300, height - 30), 
                            (width//2 + 300, height - 10), (255, 255, 255), 2)
                
                # Eye detection status
                if eye_metrics['detected']:
                    status_text = "👁️ Eyes: DETECTED"
                    color = (100, 255, 100)
                else:
                    status_text = "👁️ Eyes: SEARCHING..."
                    color = (100, 100, 255)
                
                cv2.putText(processed_frame, status_text, (width - 250, height - 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Show frame
                cv2.imshow("AI Vision Screening", processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('y'):  # Yes, can see clearly
                    response = {
                        'test_type': test_key,
                        'test_name': test_info['name'],
                        'test_description': f"{test_info['name']}: {pattern_description}",
                        'user_answer': 'Y',
                        'timestamp': time.time() - test_start_time,
                        'difficulty': (test_index + 1) / max_patterns,
                        'pattern_index': test_index
                    }
                    test_responses.append(response)
                    print(f"   ✅ Response: Yes (Clear vision)")
                    
                    # Move to next pattern in this test
                    test_info['current_index'] += 1
                    time.sleep(0.3)  # Brief pause
                    break
                
                elif key == ord('n'):  # No, blurry
                    response = {
                        'test_type': test_key,
                        'test_name': test_info['name'],
                        'test_description': f"{test_info['name']}: {pattern_description}",
                        'user_answer': 'N',
                        'timestamp': time.time() - test_start_time,
                        'difficulty': (test_index + 1) / max_patterns,
                        'pattern_index': test_index
                    }
                    test_responses.append(response)
                    print(f"   ❌ Response: No (Blurry vision)")
                    
                    # Move to next pattern
                    test_info['current_index'] += 1
                    time.sleep(0.3)
                    break
                
                elif key == ord('s'):  # Skip to next test
                    print(f"   ⏭️ Skipping to next test")
                    test_info['current_index'] = 0
                    current_test_index += 1
                    break
                
                elif key == ord('r'):  # Repeat current pattern
                    print(f"   🔄 Repeating pattern")
                    # Don't advance index, just break to restart same pattern
                    break
                
                elif key == ord('q'):  # Quit and generate report
                    print("\n🛑 User requested to quit and generate report")
                    current_test_index = len(test_sequence)  # Exit loop
                    break
                
                elif key == 27:  # ESC key
                    print("\n🛑 ESC pressed, quitting...")
                    current_test_index = len(test_sequence)
                    break
                
                frame_count += 1
        
        # Generate final report
        print("\n" + "=" * 70)
        print("📊 GENERATING COMPREHENSIVE REPORT...")
        
        # Analyze results
        analysis_results = screener.analyze_results(test_responses)
        
        # Generate graphs
        print("📈 Generating graphs...")
        graphs_data = screener.generate_graphs(analysis_results, patient_info['name'])
        
        # Save report
        print("💾 Saving report...")
        report_file, graphs = screener.save_report(analysis_results, graphs_data, patient_info)
        
        # Display results on screen
        results_screen = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(results_screen, "🎉 VISION SCREENING COMPLETE!", 
                   (width//2 - 300, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Results
        results_text = [
            f"Overall Score: {analysis_results['overall_score']:.1f}%",
            f"Vision Condition: {analysis_results['condition']}",
            f"Glasses Needed: {'YES' if analysis_results['glasses_needed'] else 'NO'}",
            "",
            "📁 Report saved:",
            f"  {os.path.basename(report_file)}",
            "",
            "📊 Graphs generated:"
        ]
        
        for i, (graph_name, _) in enumerate(graphs):
            results_text.append(f"  ✓ {graph_name}")
        
        results_text.extend([
            "",
            "Press any key to view graphs...",
            "Press 'ESC' to exit"
        ])
        
        # Display results
        y_pos = 150
        for i, line in enumerate(results_text):
            color = (255, 255, 255)
            if "Overall Score" in line:
                if analysis_results['overall_score'] >= 70:
                    color = (0, 255, 0)
                elif analysis_results['overall_score'] >= 50:
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)
            elif "YES" in line:
                color = (0, 0, 255)
            elif "NO" in line:
                color = (0, 255, 0)
            
            cv2.putText(results_screen, line, 
                       (width//2 - 300, y_pos + i*40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2 if i < 3 else 1)
        
        cv2.imshow("AI Vision Screening", results_screen)
        
        # Wait for key press
        key = cv2.waitKey(0)
        
        # Show generated graphs if user wants
        if key != 27:  # Not ESC
            for graph_name, graph_path in graphs:
                img = cv2.imread(graph_path)
                if img is not None:
                    cv2.imshow(graph_name, img)
                    cv2.waitKey(1000)  # Show each graph for 1 second
        
        print("\n" + "=" * 70)
        print("🎉 VISION SCREENING COMPLETE!")
        print("=" * 70)
        print(f"\n📈 YOUR RESULTS:")
        print(f"   Overall Score: {analysis_results['overall_score']:.1f}%")
        print(f"   Condition: {analysis_results['condition']}")
        print(f"   Glasses Needed: {'YES' if analysis_results['glasses_needed'] else 'NO'}")
        print(f"\n💡 RECOMMENDATION:")
        print(f"   {analysis_results['glasses_recommendation']}")
        print(f"\n📁 Reports saved in: {screener.results_dir}/")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ System shutdown complete.")

if __name__ == "__main__":
    main()