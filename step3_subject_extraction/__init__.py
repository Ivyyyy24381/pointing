"""
Step 3: Subject Extraction (Optional)

Detect subjects (dog/baby) in lower half of image.
"""

from .subject_detector import SubjectDetector, SubjectDetectionResult

__all__ = ['SubjectDetector', 'SubjectDetectionResult']
