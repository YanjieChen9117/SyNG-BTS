Methods
=====

.. _apply:

ApplyExperiment
----------------

.. autofunction:: syng_bts_imports.python.Experiments_new.ApplyExperiment

For example:

>>> # running application on case study BRCASubtype
>>> ApplyExperiment(path = "../Case/BRCASubtype/", dataname = "BRCASubtypeSel_train", 
>>>                 apply_log = True, new_size = [1000], model = "CVAE1-20",
>>>                 batch_frac = 0.1, learning_rate = 0.0005, epoch = 10,
>>>                 early_stop_num = 30, off_aug = None, AE_head_num = 2,
>>>                 Gaussian_head_num = 9, pre_model = None, save_model = None)
       

.. _pilot:

PilotExperiment
------------

.. autofunction:: syng_bts_imports.python.Experiments_new.PilotExperiment

For example:

>>> PilotExperiment(dataname = "SKCMPositive_4", pilot_size = [100],
>>>                 model = "VAE1-10", batch_frac = 0.1, 
>>>                 learning_rate = 0.0005, pre_model = None,
>>>                 epoch = None,  off_aug = None, early_stop_num = 30,
>>>                 AE_head_num = 2, Gaussian_head_num = 9)


.. _transfer:

TransferExperiment
----------------

.. autofunction:: syng_bts_imports.python.Experiments_new.TransferExperiment

For example:

>>> TransferExperiment(pilot_size = None, fromname = "PRAD", toname = "BRCA", fromsize = 551, 
>>>                    new_size = 500, apply_log = True, model = "maf", epoch = 10,
>>>                    batch_frac = 0.1, learning_rate = 0.0005, off_aug = None)