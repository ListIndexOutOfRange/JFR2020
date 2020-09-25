""" GENERIC CONFIGURATION FILE FOR TRAINING PIPELINE


- In all the code we call a model the set of:
    1. a network
    2. an optimizer
    3. a scheduler
    4. a criterion (a loss)
    
    This is consistant with the Pytorch Lightning philosophy.
    This is somewhat unusual and one must remind the definition of model and net here.


- Thus this config file is organised in the following way:
  5 dataclasses are defined:

    +-------------------------------------------------------------------------------------+
    | Dataloader  |            used to instanciate a DicomDataModule (see data.py)        |
    +-------------+-----------------------------------------------------------------------+
    | Network     |           Concatenated into a Model config meta dataclass             |
    | Optimizer   |         (basically a wrapper, according to our terminology).          |
    | Scheduler   |        This Model config dataclass will be used to instanciate        |   
    | Criterion   |                 a LightningModel object (model.py)                    |
    +-------------+-----------------------------------------------------------------------+

****
Note on implementation:

Since lists and dicts are mutable objects in Python, they should not be used as default
arguments of class (as by doing so, two instances of the same class will share the same 
mutable object).
Thus it is not possible to set a mutable default argument.
However, we think in this context we can safely do it.
That's why we use a little workaround in order to so, by using the fied function from the
dataclasses standart library.

One can assume that in this context:
> default_list_attribute = field(default_factory = lambda: the_default_list)
is equivalent to:
> default_list_attribute = the_default_list
"""




from dataclasses import dataclass, field

# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                   PREPROCESSING CONFIG                              | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

@dataclass
class Preprocess:

    """ Defines input/output paths for preprocessing json/nifti data and parameters for 
        mask creation and cropping. 

        - max_depth: since several nifti files are associated with one json file, 
                     we keep the biggest one with respect to max_depth, that is 
                     the biggest available with less slices than max_depth. 
        
        - cube_side: when making a mask from an annotation, we 'draw' a white cube around it.
                     Here one can specify this cube side length.
                     This will be used to process a logical AND with a threshold mask.

        - factor: when removing useless slices we delete the ones with a mean intensity too 
                  far from the scan standart deviation. This factor allows to control the severity
                  with which one slice is saw as an outlier.

        - margin: since we don't want annotations (ie white pixels) to be on the edge of masks,
                  we define a margin to add slices on the edge of mask if needed.

        - target_depth: since each scan (and mask) has a unique depth, from each scan we generate
                        a certain number of cubes of depth equals to target_depth by padding+cutting
                        a given scan.

        - steps: list of int or str specifying which preprocessing steps to perform:
                 1. selecting couples (json_path, nifti_path) and making and saving masks.
                 2. cropping and saving scans and masks (3d).
                 3. creating fixed size cube by doing z padding -> z cutting -> x,y cutting.
                 4. augment data
        
        - augment_factor: increase the dataset size by a factor of augment_factor.

        - augment_proba: an augmentation policy is a composition of several augmentation functions,
                         applied with a probability augment_proba.
                         e.g if augment_proba=0.5, 50% of the inputed scan will be left unchanged by
                         augmentation.
    """
    input_dir:       str = "/your/path/here/sficv/"
    output_dir:      str = "/your/path/here/preprocessed/"
    max_depth:       int = 500
    cube_side:       int = 10
    factor:          int = 2
    margin:          int = 10
    target_depth:    int = 64
    augment_factor:  int = 20
    augment_proba: float = 0.8 
    steps: list = field(default_factory = lambda: [1,2,3])




# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                   DATALOADER CONFIG                                 | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

@dataclass
class Dataloader:
    """ Basic Pytorch Dataloader configuration
    
    - Batch sizes can be found automatically by Lightning (ie the largest size that fits into memory)
    by using --auto_scale_batch_size binsearch.
    Note that this doesn't work while training across multiple GPUs

    - Num Workers should be 4*(nb GPUs). 
    """
    
    scan_rootdir: str = '/your/path/here/augmented/scans/'
    mask_rootdir: str = '/your/path/here/augmented/masks/'
    train_batch_size: int = 3
    val_batch_size:   int = 3
    num_workers:      int = 4




# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                     NETWORK CONFIG                                  | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

@dataclass
class Network:
    """ Defines a network and several training tweaks.
    
    Args:
        net (str):  defines the network to train.

        activation (str): the activation function to use. Can be one of:
                        * ReLU
                        * Swish
                        * Mish
                        Mish seems to often gives best results.
                        Be careful to adjust the learning rate accordingly.
                        One can use the learning rate finder option for Pytorch Lightning.

        self_attention (bool): boolean to control the insertion of a simple self attention layer
                            inside the basic blocks of the network.
                            See https://github.com/sdoria/SimpleSelfAttention
                            Be careful to adjust the learning rate accordingly.
                            One can use the learning rate finder option for Pytorch Lightning.

        attention_sym (bool): force or not the attention matrix to be symetric.
                            Having a symetric matrix help convergence.

        shakedrop (bool) : add shakedrop at the end of the forward method
    """

    net: str = 'densenet172'
    activation: str = 'relu'
    self_attention: bool = False
    attention_sym: bool = False
    shakedrop: bool = False


# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                    OPTIMIZER CONFIG                                 | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

@dataclass
class Optimizer:
    """
    Args:
        type (str): optimizer to use. Can be one of
                    * SGD: the old robust optimizer
                    * RAlamb: stands for RAdam + LARS
                    RAdam stands for Rectified Adam

        params (dict): the optimizer's params. Must be a dict with the params named
                    exactly as in PyTorch implementation.

        use_lookahead: boolean controlling the use of LookAhead.
                    Using LookAhead with RAlamb usually works well.
                    See RangerLARS.

        lookahead: lookahead params. Must be a dict.
    """

    optim: str = 'SGD'
    params: dict = field(default_factory = lambda: {
        'SGD':    {'lr': 0.0001, 'momentum': 0.9, 'nesterov': True, 'weight_decay': 5e-4},
        'RAlamb': {'lr': 1e-3,  'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0}
    })
    use_lookahead: bool = False
    lookahead: dict = field(default_factory = lambda: {'alpha': 0.5, 'k': 6})




# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                    SCHEDULER CONFIG                                 | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

@dataclass
class Scheduler:
    """
    Args:

        type (str): scheduler to use. Can be one of
                    * ROP: Reduce LR on Plateau when a given metric stop improving.
                    Default: reduce when test loss stop decreasing.
                    * MultiStep: classic scheduler: multiply the LR at given milestones
                    by a given gamma.
                    * Cosine: Anneals the LR following a decreasing cosine curve.
                    Be careful when using Decay: the arg epochs specifies in how many
                    epochs should the annealing occurs. If for instance the total epochs
                    number is 300 and a decay of 150 is set, the cosine annealing will occurs
                    in the last 150 epochs, thus the epochs params should be set at 150
                    and not 300.
                    * WarmupCosine: Cosine Annealing but with a warmup at the beginning.
                    The LR will groth during a given number of epoch
                    * WarmRestartCosine: Same as WarmupCosine but several warmup phases occur
                    during training instead of only one at the beginning.

        params (dict): the scheduler's params. Must be a dict with the params named
                    exactly as in PyTorch implementation.

        use_delay: boolean controlling the use of Decay.
                If True, the scheduler defined will start being active after the given
                number of epochs.
                This is often usefull when using cosine or exponential annealing.

        delay: delay params. Must be a dict.
    """

    scheduler: str = 'Cosine'
    params: dict = field(default_factory = lambda: {
        'ROP'               : {'mode': 'min', 'factor': 0.2, 'patience': 20, 'verbose': True},
        'MultiStep'         : {'milestones': [120, 200], 'gamma': 0.1, 'last_epoch': -1},
        'Cosine'            : {'epochs': 300, 'eta_min': 0, 'last_epoch': -1},
        'WarmRestartsCosine': {'T_0': 150, 'T_mult': 1, 'eta_min': 0, 'last_epoch': -1}
    })
    use_warmup: bool = True
    warmup: dict = field(default_factory = lambda: {
        'multiplier': 1000, 'warmup_epochs': 5})
    use_delay: bool = False
    delay: dict = field(default_factory = lambda: {
        'delay_epochs': 150, 'after_scheduler': 'Cosine'})




# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                      LOSS CONFIG                                    | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

@dataclass
class Criterion:
    """ For now, does nothing """


# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                     MODEL CONFIG                                    | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

@dataclass
class Model:

    """
    As we said before, we call a model the set of:
        - a network
        - an optimizer
        - a learning rate scheduler
        - a criterion
    Hence this config class.
    """

    network: Network()
    optimizer: Optimizer()
    scheduler: Scheduler()
    criterion: Criterion()