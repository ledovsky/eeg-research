import numpy as np


def get_events(df_egi, n_epoches=None, epoch_types=None):
    """build events array in mne style
    event_sample | previous_event_id | event_id
    previous_event_id is not set (equals 0)

    Args:
        df_egi (pd.DataFrame)
            result of raw.to_data_frame()
        n_epoches (int)
            expected number of epoches (just simple checkup)

    Returns:
        events (np.array)
    """

    epoc_time_idx = df_egi[df_egi['epoc'] == 1].index.tolist()
    assert len(epoc_time_idx) == n_epoches
    rows = []

    if epoch_types is None:
        rows.append((epoc_time_idx[0], 0, 0))
        rows.append((epoc_time_idx[1], 0, 1))
        rows.append((epoc_time_idx[2], 0, 0))
    else:
        for i in range(len(epoch_types)):
            rows.append((epoc_time_idx[i], 0, epoch_types[i]))

    events = np.array(rows)
    return events


def get_epoches(df_egi, picks, events):
    """This function adds epoch to index
    """

    df = df_egi.iloc[:, picks].copy()

    for i in range(events.shape[0]):

        t_min = events[i, 0]
        cond = (df.index >= t_min)

        if i != (events.shape[0] - 1):
            t_max = events[i + 1, 0]
            cond &= (df.index < t_max)

        df.loc[cond, 'condition'] = events[i, 2]
        df.loc[cond, 'epoch'] = i

    df = df.reset_index().set_index(['epoch', 'condition', 'time'])
    return df


def get_epoch(df_epochs, epoch_id):
    df_epoc = df_epochs.xs(epoch_id, level='epoch')
    df_epoc.index = df_epoc.index.droplevel('condition')
    return df_epoc

