from sklearn.model_selection import train_test_split

def split_by_engine(df, test_size=0.2, random_state=42):
    engine_ids = df['unit'].unique()
    
    train_ids, val_ids = train_test_split(
        engine_ids,
        test_size=test_size,
        random_state=random_state
    )
    
    train_df = df[df['unit'].isin(train_ids)]
    val_df = df[df['unit'].isin(val_ids)]
    
    return train_df, val_df