def wo_justice_label_filter(label):
    return not ('Acquit' in label
                or 'Appeal' in label
                or 'Arrest-Jail' in label
                or 'Charge-Indict' in label
                or 'Convict' in label
                or 'Appeal' in label
                or 'Execute' in label
                or 'Fine' in label
                or 'Pardon' in label
                or 'Release-Parole' in label
                or 'Sentence' in label
                or 'Sue' in label
                or 'Trial-Hearing' in label
                or 'Extradite' in label)

def wo_business_label_filter(label):
    return not ('Declare-Bankruptcy' in label or 'End-Org' in label or 'Merge-Org' in label or 'Start-Org' in label)

def wo_contacts_label_filter(label):
    return not ('Meet' in label or 'Phone-Write' in label)

def wo_movement_label_filter(label):
    return 'Transport' not in label

def only_justice_label_filter(label):
    return ('Acquit' in label
                or 'Appeal' in label
                or 'Arrest-Jail' in label
                or 'Charge-Indict' in label
                or 'Convict' in label
                or 'Appeal' in label
                or 'Execute' in label
                or 'Fine' in label
                or 'Pardon' in label
                or 'Release-Parole' in label
                or 'Sentence' in label
                or 'Sue' in label
                or 'Trial-Hearing' in label
                or 'Extradite' in label)

def only_business_label_filter(label):
    return 'Declare-Bankruptcy' in label or 'End-Org' in label or 'Merge-Org' in label or 'Start-Org' in label

def only_contacts_label_filter(label):
    return 'Meet' in label or 'Phone-Write' in label

def only_movement_label_filter(label):
    return 'Transport' in label