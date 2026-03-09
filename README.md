# 100 NumPy Exercises - Learning Journey 🚀

## 📌 Overview
This repository documents my progress through the classic "100 NumPy Exercises" challenge. I am tackling these problems to build a rock-solid foundation in numerical computing, array manipulation, and data handling in Python. 

## 🎯 Motivation
As a computer applications student working towards a career in Artificial Intelligence and Machine Learning, mastering NumPy is a critical step. Instead of rushing, I am solving and uploading these in batches of 10 per day to build a consistent coding habit and ensure I deeply understand concepts like broadcasting, vectorization, and advanced indexing.

## 🛠️ Tech Stack
* **Language:** Python 3.13.7
* **Library:** NumPy
* **Environment:** VS Code

## 📈 Progress Tracker (A 10-Day Streak)
I update this list daily as I push new solutions.

- [x] **Day 1:** Questions 1 - 10 (Basics & Array Creation)
- [x] **Day 2:** Questions 11 - 20 (Indexing & Slicing)
- [x] **Day 3:** Questions 21 - 30 (Feature Scaling & Matrix Math)
- [x] **Day 4:** Questions 31 - 40 (Temporal Data & In-Place Operations)
- [x] **Day 5:** Questions 41 - 50 (Under the Hood & Hardware Limits)
- [x] **Day 6:** Questions 51 - 60 (Spatial Distances & Memory Views)
- [x] **Day 7:** Questions 61 - 70 (Accumulation & Einstein Summation)
- [x] **Day 8:** Questions 71 - 80 (Sliding Windows & Dimensionality)
- [x] **Day 9:** Questions 81 - 90 (Pooling operations & Top-K Sampling)
- [x] **Day 10:** Questions 91 - 100 (Advanced Mastery)

## 💡 Key Takeaways
*(I will update this section with interesting functions or concepts I discover along the way).*
* **Day 1:** Learned the difference between `np.zeros()` and `np.empty()`.
* **Day 2:** Discovered that `0.3 == 3 * 0.1` is actually `False` in Python due to floating-point precision! In ML, it is much safer to use `np.isclose()` to compare decimal tensors.
* **Day 3:** Learned how to implement Z-score standardization for model inputs and why pure Python logical operators (like in Z < Z > Z) cause an ambiguous truth value error in NumPy.
* **Day 4:** Solved Q31-40 with a focus on memory-efficient array operations and temporal data handling"
* **Day 5:** Reached the halfway point! Explored how np.add.reduce bypasses function overhead for faster execution, and mapped data type hardware limits (crucial for model quantization).
* **Day 6:** Mastered using .view() to perform zero-copy, in-place memory casts between floats and integers, a critical technique for edge-device memory optimization.
* **Day 7:** Discovered np.einsum for highly optimized tensor operations, bypassing the massive memory waste of computing full intermediate dot products.
* **Day 8:** Utilized sliding_window_view for memory-safe rolling operations, a core technique for parsing continuous streams in Natural Language Processing.
* **Day 8:** Utilized sliding_window_view for memory-safe rolling operations, a core technique for parsing continuous streams in Natural Language Processing.
* **Day 9:** Implemented np.argpartition for linear-time Top-K extraction, which is the exact mathematical foundation for decoding algorithms in Large Language Models.
* **Day 10:** Challenge complete. Mastered integer-to-binary matrix conversions (the foundation of AI model quantization) and statistical bootstrapping for model confidence evaluation.

## 🤝 Acknowledgments
Questions sourced from the community-maintained [100 numpy exercises](https://github.com/rougier/numpy-100) repository.
