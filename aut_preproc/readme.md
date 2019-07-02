# Preprocessing of autists dataset

There are different set of channels (even EEG) and different reference for different records.

For most records channel name is `X-A1` or `X-A2`, were `X` is a channel name. 
There is a function that changes reference to `A1` and normalize names to `X`.
Also channel `A2` is added and also `A1` with zero values.


