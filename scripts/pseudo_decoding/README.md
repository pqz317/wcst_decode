## Manual for feature decoding using pseudo populations
Below we document the instructions and data requirements needed to run decoding of selected features with pseudo populations

### Main script: 
`scripts/pseudo_decoding/decode_features_with_pseudo.py`

Runs decoding for each feature dimension, stores results. 

**Data requirements:**

There exists a Dataframe of sessions, located by `SESSIONS_PATH` with: 
- one column `session_name` of `str`, specifying the identifier of the session

For each session, there exists: 

A Dataframe of behavior, located by `SESS_BEHAVIOR_PATH`, with columns: 
- `TrialNumber` as an int identifier for trials
- `Response`, of `str`, with at least `Correct` and `Incorrect` values. 
-  `ItemChosen`: an int from 0 - 3, indicating which card was chosen for the trial
- `Item{0/1/2/3}{Color/Shape/Pattern}`: indicating the color/shape/pattern of each of the 0 through 3 cards

A Dataframe of spikes, with pre-aligned, pre-binned spikes and firing rates, located by `SESS_SPIKES_PATH`, with columns:
- `TrialNumber`: int identifier of trial, maching behavior df
- `UnitID`: int identifier for unit, session-specific. NOTE: due to a hack in generating pseudo-unit IDs, this number should be < 100. 
- `TimeBins`: a bin identifier in float, specified in seconds, relative to the earliest bin. (ex if using 100ms bins, this would be. 0.0, 0.1, ...). NOTE: the plural `TimeBins` in the column name, this does not follow convention, but kind of stuck with it. 
- `SpikeCounts`: an int indicating how many spikes occured per bin NOTE: also plural. 