

This graph shows how a model's performance (measured by success rate percentage) changes as the number of training examples increases, across four different difficulty levels: easy, medium, hard, and expert.










Performance hierarchy: Easier tasks consistently achieve better performance
Easy (blue): reaches ~52% success
Medium (orange): reaches ~36-38% success
Hard (green): reaches ~30% success
Expert (red): plateaus at ~22% success

2. Learning curves:
All difficulty levels show improvement with more training examples
The most dramatic improvements happen between 10¹ and 10² training examples
After 10² examples, the curves generally plateau or show only minor improvements

3. Scaling behavior:
Easy tasks benefit the most from additional training data
Expert tasks show the least improvement with more data, suggesting these problems remain challenging even with more training

Plateaus:
Expert tasks plateau earliest, around 10² examples
Easy tasks continue to show slight improvements even at 10³ examples
Medium and hard tasks show intermediate behavior
This visualization suggests that while more training data helps across all difficulty levels, there are diminishing returns, and harder tasks remain challenging even with substantial training data.



Batch=1:

| Epoch | Attempted tasks | pred acc | comp acc | easy | medium | hard | expert |
| ----- | --------------- | -------- | -------- | ---- | ------ | ---- | ------ |
| 1     | 400             | 205/419  | 191/400  | 105  | 59     | 21   | 6      |
| 2     | 400             | 207/419  | 191/400  | 102  | 62     | 21   | 6      |
| 3     | 400             | 203/419  | 187/400  | 102  | 57     | 22   | 6      |



Batch=2:

| Epoch | Attempted tasks | pred acc | comp acc | easy | medium | hard | expert |
| ----- | --------------- | -------- | -------- | ---- | ------ | ---- | ------ |
| 1     | 400             | 208/419  | 193/400  | 104  | 62     | 20   | 7      |
| 2     | 400             | 211/419  | 195/400  | 105  | 61     | 23   | 6      |
| 3     | 400             | 214/419  | 198/400  | 108  | 61     | 23   | 6      |


Batch=3:

| Epoch | Attempted tasks | pred acc | comp acc | easy | medium | hard | expert |
| ----- | --------------- | -------- | -------- | ---- | ------ | ---- | ------ |
| 1     | 400             | 201/419  | 186/400  | 96   | 62     | 22   | 6      |
| 2     | 400             | 207/419  | 192/400  | 102  | 63     | 21   | 6      |
| 3     | 400             | 211/419  | 196/400  | 102  | 66     | 22   | 6      |
