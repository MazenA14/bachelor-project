import os
import glob
import re

directories = [
    r'c:\Users\Mazen\OneDrive - GUC\GUC\Years\Year 4\Semester 8\bachelor-project\src',
    r'c:\Users\Mazen\OneDrive - GUC\GUC\Years\Year 4\Semester 8\bachelor-project\src-extra-variables'
]

replacements = {
    # Targets
    r"y_train = train\['Gold_Close_LogReturn'\]": r"y_train = train['Silver_Close_LogReturn']",
    r"y_val = val\['Gold_Close_LogReturn'\]": r"y_val = val['Silver_Close_LogReturn']",
    r"y_test = test\['Gold_Close_LogReturn'\]": r"y_test = test['Silver_Close_LogReturn']",
    r"y_train_a = train_a\['Gold_Close_LogReturn'\]": r"y_train_a = train_a['Silver_Close_LogReturn']",
    r"y_val_a = val_a\['Gold_Close_LogReturn'\]": r"y_val_a = val_a['Silver_Close_LogReturn']",
    r"y_train_b = train_b\['Gold_Close_Residual'\]": r"y_train_b = train_b['Silver_Close_Residual']",
    r"y_val_b = val_b\['Gold_Close_Residual'\]": r"y_val_b = val_b['Silver_Close_Residual']",

    # Mathematical Reversals
    r"df_master\.loc\[X_test_hybrid\.index, 'Gold_Close'\]": r"df_master.loc[X_test_hybrid.index, 'Silver_Close']",
    r"df_master\.loc\['2024-12-31', 'Gold_Close'\]": r"df_master.loc['2024-12-31', 'Silver_Close']",
    r"df_master\.loc\[X_test_a\.index, 'Gold_Close'\]": r"df_master.loc[X_test_a.index, 'Silver_Close']",
    r"df_master\.loc\[X_test_b\.index, 'Gold_Close'\]": r"df_master.loc[X_test_b.index, 'Silver_Close']",
    r"df_master\.loc\[TRAIN_VAL_END, 'Gold_Close'\]": r"df_master.loc[TRAIN_VAL_END, 'Silver_Close']",
    r"df_b\.loc\[X_test_b\.index, 'Gold_Close_Trend'\]": r"df_b.loc[X_test_b.index, 'Silver_Close_Trend']",

    # Actual Prices & ARIMA
    r"df_master\.loc\[test\.index, 'Gold_Close'\]": r"df_master.loc[test.index, 'Silver_Close']",
    r"df_master\.loc\[test_a\.index, 'Gold_Close'\]": r"df_master.loc[test_a.index, 'Silver_Close']",
    r"df_master\['Gold_Close'\]\[:VAL_END\]": r"df_master['Silver_Close'][:VAL_END]",
    r"df_master\['Gold_Close'\]\[:TRAIN_VAL_END\]": r"df_master['Silver_Close'][:TRAIN_VAL_END]",
    r"df_master\['Gold_Close'\]\[TEST_START:\]": r"df_master['Silver_Close'][TEST_START:]",

    # Plot Labels
    r"'Actual Gold Price'": r"'Actual Silver Price'",
    r"'Gold Price \(USD\)'": r"'Silver Price (USD)'",
    r"'Gold Forecast:": r"'Silver Forecast:",
    r"'Gold Price Forecast:": r"'Silver Price Forecast:",
    r"Actual Gold Price": r"Actual Silver Price",
    r"Gold Price \(USD\)": r"Silver Price (USD)",
    
    # Text Prints
    r"Gold Price Forecast": r"Silver Price Forecast",
}

for d in directories:
    files = glob.glob(os.path.join(d, '*.py'))
    for f in files:
        # Ignore feature engineering scripts, just to be safe, though no harm done 
        # as long as we only replace specific usages of Gold as Target/Reversals.
        # Actually it's safe to process all scripts because replacements are highly targeted.
        
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
            
        new_content = content
        for pattern, replacement in replacements.items():
            new_content = re.sub(pattern, replacement, new_content)
            
        if content != new_content:
            with open(f, 'w', encoding='utf-8') as file:
                file.write(new_content)
            print(f"Updated: {f}")
print("Replacement script complete.")
