Ʀ
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
:
Maximum
x"T
y"T
z"T"
Ttype:

2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
:
Minimum
x"T
y"T
z"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
3
Square
x"T
y"T"
Ttype:
2
	
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8??
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
: *
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
: *
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:@*
dtype0
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?* 
shared_nameconv2d_2/kernel
|
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*'
_output_shapes
:@?*
dtype0
s
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_2/bias
l
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes	
:?*
dtype0
?
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??* 
shared_nameconv2d_3/kernel
}
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*(
_output_shapes
:??*
dtype0
s
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_3/bias
l
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes	
:?*
dtype0
?
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??* 
shared_nameconv2d_4/kernel
}
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*(
_output_shapes
:??*
dtype0
s
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_4/bias
l
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes	
:?*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	?*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d/kernel/m
?
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
: *
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_1/kernel/m
?
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*'
shared_nameAdam/conv2d_2/kernel/m
?
*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*'
_output_shapes
:@?*
dtype0
?
Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/conv2d_2/bias/m
z
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*'
shared_nameAdam/conv2d_3/kernel/m
?
*Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/m*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/conv2d_3/bias/m
z
(Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*'
shared_nameAdam/conv2d_4/kernel/m
?
*Adam/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/m*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/conv2d_4/bias/m
z
(Adam/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	?*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d/kernel/v
?
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
: *
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_1/kernel/v
?
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*'
shared_nameAdam/conv2d_2/kernel/v
?
*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*'
_output_shapes
:@?*
dtype0
?
Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/conv2d_2/bias/v
z
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*'
shared_nameAdam/conv2d_3/kernel/v
?
*Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/v*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/conv2d_3/bias/v
z
(Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*'
shared_nameAdam/conv2d_4/kernel/v
?
*Adam/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/v*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/conv2d_4/bias/v
z
(Adam/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	?*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    

NoOpNoOp
?R
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*?R
value?QB?Q B?Q
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
		optimizer

trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
 
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
layer_with_weights-3
layer-8
layer-9
layer_with_weights-4
layer-10
layer-11
layer-12
layer_with_weights-5
layer-13
trainable_variables
	variables
regularization_losses
 	keras_api

!	keras_api

"	keras_api

#	keras_api

$	keras_api

%	keras_api
?
&iter

'beta_1

(beta_2
	)decay
*learning_rate+m?,m?-m?.m?/m?0m?1m?2m?3m?4m?5m?6m?+v?,v?-v?.v?/v?0v?1v?2v?3v?4v?5v?6v?
V
+0
,1
-2
.3
/4
05
16
27
38
49
510
611
V
+0
,1
-2
.3
/4
05
16
27
38
49
510
611
 
?
7non_trainable_variables

trainable_variables

8layers
9metrics
:layer_metrics
	variables
regularization_losses
;layer_regularization_losses
 
 
h

+kernel
,bias
<trainable_variables
=	variables
>regularization_losses
?	keras_api
R
@trainable_variables
A	variables
Bregularization_losses
C	keras_api
R
Dtrainable_variables
E	variables
Fregularization_losses
G	keras_api
h

-kernel
.bias
Htrainable_variables
I	variables
Jregularization_losses
K	keras_api
R
Ltrainable_variables
M	variables
Nregularization_losses
O	keras_api
h

/kernel
0bias
Ptrainable_variables
Q	variables
Rregularization_losses
S	keras_api
R
Ttrainable_variables
U	variables
Vregularization_losses
W	keras_api
h

1kernel
2bias
Xtrainable_variables
Y	variables
Zregularization_losses
[	keras_api
R
\trainable_variables
]	variables
^regularization_losses
_	keras_api
h

3kernel
4bias
`trainable_variables
a	variables
bregularization_losses
c	keras_api
R
dtrainable_variables
e	variables
fregularization_losses
g	keras_api
R
htrainable_variables
i	variables
jregularization_losses
k	keras_api
h

5kernel
6bias
ltrainable_variables
m	variables
nregularization_losses
o	keras_api
V
+0
,1
-2
.3
/4
05
16
27
38
49
510
611
V
+0
,1
-2
.3
/4
05
16
27
38
49
510
611
 
?
pnon_trainable_variables
trainable_variables

qlayers
rmetrics
slayer_metrics
	variables
regularization_losses
tlayer_regularization_losses
 
 
 
 
 
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEconv2d/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d_1/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d_1/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d_2/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d_2/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d_3/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d_3/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d_4/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d_4/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUE
dense/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
 
8
0
1
2
3
4
5
6
7

u0
 
 

+0
,1

+0
,1
 
?
vnon_trainable_variables
<trainable_variables

wlayers
xmetrics
ylayer_metrics
=	variables
>regularization_losses
zlayer_regularization_losses
 
 
 
?
{non_trainable_variables
@trainable_variables

|layers
}metrics
~layer_metrics
A	variables
Bregularization_losses
layer_regularization_losses
 
 
 
?
?non_trainable_variables
Dtrainable_variables
?layers
?metrics
?layer_metrics
E	variables
Fregularization_losses
 ?layer_regularization_losses

-0
.1

-0
.1
 
?
?non_trainable_variables
Htrainable_variables
?layers
?metrics
?layer_metrics
I	variables
Jregularization_losses
 ?layer_regularization_losses
 
 
 
?
?non_trainable_variables
Ltrainable_variables
?layers
?metrics
?layer_metrics
M	variables
Nregularization_losses
 ?layer_regularization_losses

/0
01

/0
01
 
?
?non_trainable_variables
Ptrainable_variables
?layers
?metrics
?layer_metrics
Q	variables
Rregularization_losses
 ?layer_regularization_losses
 
 
 
?
?non_trainable_variables
Ttrainable_variables
?layers
?metrics
?layer_metrics
U	variables
Vregularization_losses
 ?layer_regularization_losses

10
21

10
21
 
?
?non_trainable_variables
Xtrainable_variables
?layers
?metrics
?layer_metrics
Y	variables
Zregularization_losses
 ?layer_regularization_losses
 
 
 
?
?non_trainable_variables
\trainable_variables
?layers
?metrics
?layer_metrics
]	variables
^regularization_losses
 ?layer_regularization_losses

30
41

30
41
 
?
?non_trainable_variables
`trainable_variables
?layers
?metrics
?layer_metrics
a	variables
bregularization_losses
 ?layer_regularization_losses
 
 
 
?
?non_trainable_variables
dtrainable_variables
?layers
?metrics
?layer_metrics
e	variables
fregularization_losses
 ?layer_regularization_losses
 
 
 
?
?non_trainable_variables
htrainable_variables
?layers
?metrics
?layer_metrics
i	variables
jregularization_losses
 ?layer_regularization_losses

50
61

50
61
 
?
?non_trainable_variables
ltrainable_variables
?layers
?metrics
?layer_metrics
m	variables
nregularization_losses
 ?layer_regularization_losses
 
f
0
1
2
3
4
5
6
7
8
9
10
11
12
13
 
 
 
8

?total

?count
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
vt
VARIABLE_VALUEAdam/conv2d/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/conv2d/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_1/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d_1/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_2/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d_2/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_3/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d_3/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_4/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d_4/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/dense/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/conv2d/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_1/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d_1/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_2/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d_2/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_3/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d_3/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_4/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d_4/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/dense/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
serving_default_input_2Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2conv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasdense/kernel
dense/biasConstConst_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *+
f&R$
"__inference_signature_wrapper_8165
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp*Adam/conv2d_3/kernel/m/Read/ReadVariableOp(Adam/conv2d_3/bias/m/Read/ReadVariableOp*Adam/conv2d_4/kernel/m/Read/ReadVariableOp(Adam/conv2d_4/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp*Adam/conv2d_3/kernel/v/Read/ReadVariableOp(Adam/conv2d_3/bias/v/Read/ReadVariableOp*Adam/conv2d_4/kernel/v/Read/ReadVariableOp(Adam/conv2d_4/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOpConst_2*8
Tin1
/2-	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *&
f!R
__inference__traced_save_8967
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasdense/kernel
dense/biastotalcountAdam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/mAdam/conv2d_3/kernel/mAdam/conv2d_3/bias/mAdam/conv2d_4/kernel/mAdam/conv2d_4/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/vAdam/conv2d_3/kernel/vAdam/conv2d_3/bias/vAdam/conv2d_4/kernel/vAdam/conv2d_4/bias/vAdam/dense/kernel/vAdam/dense/bias/v*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *)
f$R"
 __inference__traced_restore_9106??
?
?
M__inference_sig_ver_siamese_cnn_layer_call_and_return_conditional_losses_8004

inputs
inputs_1

model_7956

model_7958

model_7960

model_7962

model_7964

model_7966

model_7968

model_7970

model_7972

model_7974

model_7976

model_7978,
(tf_clip_by_value_clip_by_value_minimum_y$
 tf_clip_by_value_clip_by_value_y
identity??model/StatefulPartitionedCall?model/StatefulPartitionedCall_1?
model/StatefulPartitionedCallStatefulPartitionedCallinputs
model_7956
model_7958
model_7960
model_7962
model_7964
model_7966
model_7968
model_7970
model_7972
model_7974
model_7976
model_7978*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_76882
model/StatefulPartitionedCall?
model/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1
model_7956
model_7958
model_7960
model_7962
model_7964
model_7966
model_7968
model_7970
model_7972
model_7974
model_7976
model_7978*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_76882!
model/StatefulPartitionedCall_1?
tf.math.subtract/SubSub&model/StatefulPartitionedCall:output:0(model/StatefulPartitionedCall_1:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract/Sub?
tf.math.square/SquareSquaretf.math.subtract/Sub:z:0*
T0*'
_output_shapes
:?????????2
tf.math.square/Square?
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2*
(tf.math.reduce_sum/Sum/reduction_indices?
tf.math.reduce_sum/SumSumtf.math.square/Square:y:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
tf.math.reduce_sum/Sum?
&tf.clip_by_value/clip_by_value/MinimumMinimumtf.math.reduce_sum/Sum:output:0(tf_clip_by_value_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????2(
&tf.clip_by_value/clip_by_value/Minimum?
tf.clip_by_value/clip_by_valueMaximum*tf.clip_by_value/clip_by_value/Minimum:z:0 tf_clip_by_value_clip_by_value_y*
T0*'
_output_shapes
:?????????2 
tf.clip_by_value/clip_by_value?
tf.math.sqrt/SqrtSqrt"tf.clip_by_value/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
tf.math.sqrt/Sqrt?
IdentityIdentitytf.math.sqrt/Sqrt:y:0^model/StatefulPartitionedCall ^model/StatefulPartitionedCall_1*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesp
n:???????????:???????????::::::::::::: : 2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model/StatefulPartitionedCall_1model/StatefulPartitionedCall_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs:YU
1
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
z
%__inference_conv2d_layer_call_fn_8659

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_73872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
a
(__inference_dropout_1_layer_call_fn_8788

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_75572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
A__inference_dropout_layer_call_and_return_conditional_losses_7416

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:??????????n 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????n *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????n 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????n 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????n 2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:??????????n 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????n :X T
0
_output_shapes
:??????????n 
 
_user_specified_nameinputs
?

?
B__inference_conv2d_1_layer_call_and_return_conditional_losses_7445

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????n@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????n@2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????n@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????n@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????n ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????n 
 
_user_specified_nameinputs
?X
?
?__inference_model_layer_call_and_return_conditional_losses_8528

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d/BiasAddw
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
conv2d/Relu?
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*0
_output_shapes
:??????????n *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPools
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/dropout/Const?
dropout/dropout/MulMulmax_pooling2d/MaxPool:output:0dropout/dropout/Const:output:0*
T0*0
_output_shapes
:??????????n 2
dropout/dropout/Mul|
dropout/dropout/ShapeShapemax_pooling2d/MaxPool:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*0
_output_shapes
:??????????n *
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????n 2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????n 2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*0
_output_shapes
:??????????n 2
dropout/dropout/Mul_1?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Ddropout/dropout/Mul_1:z:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????n@*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????n@2
conv2d_1/BiasAdd|
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????n@2
conv2d_1/Relu?
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:?????????n7@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????n7?*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????n7?2
conv2d_2/BiasAdd|
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????n7?2
conv2d_2/Relu?
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*0
_output_shapes
:?????????7?*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPool?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2D max_pooling2d_2/MaxPool:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????7?*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????7?2
conv2d_3/BiasAdd|
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????7?2
conv2d_3/Relu?
max_pooling2d_3/MaxPoolMaxPoolconv2d_3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPool?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2D max_pooling2d_3/MaxPool:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_4/BiasAdd|
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_4/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_1/dropout/Const?
dropout_1/dropout/MulMulconv2d_4/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*0
_output_shapes
:??????????2
dropout_1/dropout/Mul}
dropout_1/dropout/ShapeShapeconv2d_4/Relu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout_1/dropout/Mul_1?
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indices?
global_average_pooling2d/MeanMeandropout_1/dropout/Mul_1:z:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
global_average_pooling2d/Mean?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMul&global_average_pooling2d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAdd?
IdentityIdentitydense/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:???????????::::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
|
'__inference_conv2d_2_layer_call_fn_8726

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????n7?*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_74732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????n7?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????n7@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????n7@
 
_user_specified_nameinputs
?F
?
?__inference_model_layer_call_and_return_conditional_losses_8581

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d/BiasAddw
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
conv2d/Relu?
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*0
_output_shapes
:??????????n *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
dropout/IdentityIdentitymax_pooling2d/MaxPool:output:0*
T0*0
_output_shapes
:??????????n 2
dropout/Identity?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Ddropout/Identity:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????n@*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????n@2
conv2d_1/BiasAdd|
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????n@2
conv2d_1/Relu?
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:?????????n7@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????n7?*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????n7?2
conv2d_2/BiasAdd|
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????n7?2
conv2d_2/Relu?
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*0
_output_shapes
:?????????7?*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPool?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2D max_pooling2d_2/MaxPool:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????7?*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????7?2
conv2d_3/BiasAdd|
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????7?2
conv2d_3/Relu?
max_pooling2d_3/MaxPoolMaxPoolconv2d_3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPool?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2D max_pooling2d_3/MaxPool:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_4/BiasAdd|
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_4/Relu?
dropout_1/IdentityIdentityconv2d_4/Relu:activations:0*
T0*0
_output_shapes
:??????????2
dropout_1/Identity?
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indices?
global_average_pooling2d/MeanMeandropout_1/Identity:output:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2
global_average_pooling2d/Mean?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMul&global_average_pooling2d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAdd?
IdentityIdentitydense/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:???????????::::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
B__inference_conv2d_1_layer_call_and_return_conditional_losses_8697

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????n@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????n@2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????n@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????n@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????n ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????n 
 
_user_specified_nameinputs
?

?
B__inference_conv2d_4_layer_call_and_return_conditional_losses_7529

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
M__inference_sig_ver_siamese_cnn_layer_call_and_return_conditional_losses_7948
input_1
input_2

model_7900

model_7902

model_7904

model_7906

model_7908

model_7910

model_7912

model_7914

model_7916

model_7918

model_7920

model_7922,
(tf_clip_by_value_clip_by_value_minimum_y$
 tf_clip_by_value_clip_by_value_y
identity??model/StatefulPartitionedCall?model/StatefulPartitionedCall_1?
model/StatefulPartitionedCallStatefulPartitionedCallinput_1
model_7900
model_7902
model_7904
model_7906
model_7908
model_7910
model_7912
model_7914
model_7916
model_7918
model_7920
model_7922*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_77582
model/StatefulPartitionedCall?
model/StatefulPartitionedCall_1StatefulPartitionedCallinput_2
model_7900
model_7902
model_7904
model_7906
model_7908
model_7910
model_7912
model_7914
model_7916
model_7918
model_7920
model_7922*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_77582!
model/StatefulPartitionedCall_1?
tf.math.subtract/SubSub&model/StatefulPartitionedCall:output:0(model/StatefulPartitionedCall_1:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract/Sub?
tf.math.square/SquareSquaretf.math.subtract/Sub:z:0*
T0*'
_output_shapes
:?????????2
tf.math.square/Square?
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2*
(tf.math.reduce_sum/Sum/reduction_indices?
tf.math.reduce_sum/SumSumtf.math.square/Square:y:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
tf.math.reduce_sum/Sum?
&tf.clip_by_value/clip_by_value/MinimumMinimumtf.math.reduce_sum/Sum:output:0(tf_clip_by_value_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????2(
&tf.clip_by_value/clip_by_value/Minimum?
tf.clip_by_value/clip_by_valueMaximum*tf.clip_by_value/clip_by_value/Minimum:z:0 tf_clip_by_value_clip_by_value_y*
T0*'
_output_shapes
:?????????2 
tf.clip_by_value/clip_by_value?
tf.math.sqrt/SqrtSqrt"tf.clip_by_value/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
tf.math.sqrt/Sqrt?
IdentityIdentitytf.math.sqrt/Sqrt:y:0^model/StatefulPartitionedCall ^model/StatefulPartitionedCall_1*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesp
n:???????????:???????????::::::::::::: : 2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model/StatefulPartitionedCall_1model/StatefulPartitionedCall_1:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1:ZV
1
_output_shapes
:???????????
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: 
?
J
.__inference_max_pooling2d_1_layer_call_fn_7335

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_73292
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
?
$__inference_model_layer_call_fn_7785
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_77582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:???????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_3
?Z
?
__inference__traced_save_8967
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableop5
1savev2_adam_conv2d_3_kernel_m_read_readvariableop3
/savev2_adam_conv2d_3_bias_m_read_readvariableop5
1savev2_adam_conv2d_4_kernel_m_read_readvariableop3
/savev2_adam_conv2d_4_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop3
/savev2_adam_conv2d_3_bias_v_read_readvariableop5
1savev2_adam_conv2d_4_kernel_v_read_readvariableop3
/savev2_adam_conv2d_4_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop
savev2_const_2

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*?
value?B?,B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_3_bias_m_read_readvariableop1savev2_adam_conv2d_4_kernel_m_read_readvariableop/savev2_adam_conv2d_4_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop/savev2_adam_conv2d_3_bias_v_read_readvariableop1savev2_adam_conv2d_4_kernel_v_read_readvariableop/savev2_adam_conv2d_4_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableopsavev2_const_2"/device:CPU:0*
_output_shapes
 *:
dtypes0
.2,	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : : : : @:@:@?:?:??:?:??:?:	?:: : : : : @:@:@?:?:??:?:??:?:	?:: : : @:@:@?:?:??:?:??:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 	

_output_shapes
:@:-
)
'
_output_shapes
:@?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::, (
&
_output_shapes
: : !

_output_shapes
: :,"(
&
_output_shapes
: @: #

_output_shapes
:@:-$)
'
_output_shapes
:@?:!%

_output_shapes	
:?:.&*
(
_output_shapes
:??:!'

_output_shapes	
:?:.(*
(
_output_shapes
:??:!)

_output_shapes	
:?:%*!

_output_shapes
:	?: +

_output_shapes
::,

_output_shapes
: 
?	
?
$__inference_model_layer_call_fn_8610

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_76882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:???????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?6
?
?__inference_model_layer_call_and_return_conditional_losses_7644
input_3
conv2d_7606
conv2d_7608
conv2d_1_7613
conv2d_1_7615
conv2d_2_7619
conv2d_2_7621
conv2d_3_7625
conv2d_3_7627
conv2d_4_7631
conv2d_4_7633

dense_7638

dense_7640
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_7606conv2d_7608*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_73872 
conv2d/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????n * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_73172
max_pooling2d/PartitionedCall?
dropout/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????n * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_74212
dropout/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv2d_1_7613conv2d_1_7615*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????n@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_74452"
 conv2d_1/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????n7@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_73292!
max_pooling2d_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_7619conv2d_2_7621*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????n7?*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_74732"
 conv2d_2/StatefulPartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????7?* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_73412!
max_pooling2d_2/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_3_7625conv2d_3_7627*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????7?*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_75012"
 conv2d_3/StatefulPartitionedCall?
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_73532!
max_pooling2d_3/PartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_4_7631conv2d_4_7633*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_75292"
 conv2d_4/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_75622
dropout_1/PartitionedCall?
(global_average_pooling2d/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *[
fVRT
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_73662*
(global_average_pooling2d/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0
dense_7638
dense_7640*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_75862
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:???????????::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_3
?

?
B__inference_conv2d_2_layer_call_and_return_conditional_losses_8717

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????n7?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????n7?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????n7?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????n7?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????n7@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????n7@
 
_user_specified_nameinputs
??
?
M__inference_sig_ver_siamese_cnn_layer_call_and_return_conditional_losses_8393
inputs_0
inputs_1/
+model_conv2d_conv2d_readvariableop_resource0
,model_conv2d_biasadd_readvariableop_resource1
-model_conv2d_1_conv2d_readvariableop_resource2
.model_conv2d_1_biasadd_readvariableop_resource1
-model_conv2d_2_conv2d_readvariableop_resource2
.model_conv2d_2_biasadd_readvariableop_resource1
-model_conv2d_3_conv2d_readvariableop_resource2
.model_conv2d_3_biasadd_readvariableop_resource1
-model_conv2d_4_conv2d_readvariableop_resource2
.model_conv2d_4_biasadd_readvariableop_resource.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource,
(tf_clip_by_value_clip_by_value_minimum_y$
 tf_clip_by_value_clip_by_value_y
identity??#model/conv2d/BiasAdd/ReadVariableOp?%model/conv2d/BiasAdd_1/ReadVariableOp?"model/conv2d/Conv2D/ReadVariableOp?$model/conv2d/Conv2D_1/ReadVariableOp?%model/conv2d_1/BiasAdd/ReadVariableOp?'model/conv2d_1/BiasAdd_1/ReadVariableOp?$model/conv2d_1/Conv2D/ReadVariableOp?&model/conv2d_1/Conv2D_1/ReadVariableOp?%model/conv2d_2/BiasAdd/ReadVariableOp?'model/conv2d_2/BiasAdd_1/ReadVariableOp?$model/conv2d_2/Conv2D/ReadVariableOp?&model/conv2d_2/Conv2D_1/ReadVariableOp?%model/conv2d_3/BiasAdd/ReadVariableOp?'model/conv2d_3/BiasAdd_1/ReadVariableOp?$model/conv2d_3/Conv2D/ReadVariableOp?&model/conv2d_3/Conv2D_1/ReadVariableOp?%model/conv2d_4/BiasAdd/ReadVariableOp?'model/conv2d_4/BiasAdd_1/ReadVariableOp?$model/conv2d_4/Conv2D/ReadVariableOp?&model/conv2d_4/Conv2D_1/ReadVariableOp?"model/dense/BiasAdd/ReadVariableOp?$model/dense/BiasAdd_1/ReadVariableOp?!model/dense/MatMul/ReadVariableOp?#model/dense/MatMul_1/ReadVariableOp?
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02$
"model/conv2d/Conv2D/ReadVariableOp?
model/conv2d/Conv2DConv2Dinputs_0*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
model/conv2d/Conv2D?
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#model/conv2d/BiasAdd/ReadVariableOp?
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
model/conv2d/BiasAdd?
model/conv2d/ReluRelumodel/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
model/conv2d/Relu?
model/max_pooling2d/MaxPoolMaxPoolmodel/conv2d/Relu:activations:0*0
_output_shapes
:??????????n *
ksize
*
paddingVALID*
strides
2
model/max_pooling2d/MaxPool?
model/dropout/IdentityIdentity$model/max_pooling2d/MaxPool:output:0*
T0*0
_output_shapes
:??????????n 2
model/dropout/Identity?
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02&
$model/conv2d_1/Conv2D/ReadVariableOp?
model/conv2d_1/Conv2DConv2Dmodel/dropout/Identity:output:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????n@*
paddingSAME*
strides
2
model/conv2d_1/Conv2D?
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model/conv2d_1/BiasAdd/ReadVariableOp?
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????n@2
model/conv2d_1/BiasAdd?
model/conv2d_1/ReluRelumodel/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????n@2
model/conv2d_1/Relu?
model/max_pooling2d_1/MaxPoolMaxPool!model/conv2d_1/Relu:activations:0*/
_output_shapes
:?????????n7@*
ksize
*
paddingVALID*
strides
2
model/max_pooling2d_1/MaxPool?
$model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02&
$model/conv2d_2/Conv2D/ReadVariableOp?
model/conv2d_2/Conv2DConv2D&model/max_pooling2d_1/MaxPool:output:0,model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????n7?*
paddingSAME*
strides
2
model/conv2d_2/Conv2D?
%model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%model/conv2d_2/BiasAdd/ReadVariableOp?
model/conv2d_2/BiasAddBiasAddmodel/conv2d_2/Conv2D:output:0-model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????n7?2
model/conv2d_2/BiasAdd?
model/conv2d_2/ReluRelumodel/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????n7?2
model/conv2d_2/Relu?
model/max_pooling2d_2/MaxPoolMaxPool!model/conv2d_2/Relu:activations:0*0
_output_shapes
:?????????7?*
ksize
*
paddingVALID*
strides
2
model/max_pooling2d_2/MaxPool?
$model/conv2d_3/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02&
$model/conv2d_3/Conv2D/ReadVariableOp?
model/conv2d_3/Conv2DConv2D&model/max_pooling2d_2/MaxPool:output:0,model/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????7?*
paddingSAME*
strides
2
model/conv2d_3/Conv2D?
%model/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%model/conv2d_3/BiasAdd/ReadVariableOp?
model/conv2d_3/BiasAddBiasAddmodel/conv2d_3/Conv2D:output:0-model/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????7?2
model/conv2d_3/BiasAdd?
model/conv2d_3/ReluRelumodel/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????7?2
model/conv2d_3/Relu?
model/max_pooling2d_3/MaxPoolMaxPool!model/conv2d_3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
model/max_pooling2d_3/MaxPool?
$model/conv2d_4/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02&
$model/conv2d_4/Conv2D/ReadVariableOp?
model/conv2d_4/Conv2DConv2D&model/max_pooling2d_3/MaxPool:output:0,model/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model/conv2d_4/Conv2D?
%model/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%model/conv2d_4/BiasAdd/ReadVariableOp?
model/conv2d_4/BiasAddBiasAddmodel/conv2d_4/Conv2D:output:0-model/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model/conv2d_4/BiasAdd?
model/conv2d_4/ReluRelumodel/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model/conv2d_4/Relu?
model/dropout_1/IdentityIdentity!model/conv2d_4/Relu:activations:0*
T0*0
_output_shapes
:??????????2
model/dropout_1/Identity?
5model/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      27
5model/global_average_pooling2d/Mean/reduction_indices?
#model/global_average_pooling2d/MeanMean!model/dropout_1/Identity:output:0>model/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2%
#model/global_average_pooling2d/Mean?
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!model/dense/MatMul/ReadVariableOp?
model/dense/MatMulMatMul,model/global_average_pooling2d/Mean:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense/MatMul?
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/dense/BiasAdd/ReadVariableOp?
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense/BiasAdd?
$model/conv2d/Conv2D_1/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02&
$model/conv2d/Conv2D_1/ReadVariableOp?
model/conv2d/Conv2D_1Conv2Dinputs_1,model/conv2d/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
model/conv2d/Conv2D_1?
%model/conv2d/BiasAdd_1/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%model/conv2d/BiasAdd_1/ReadVariableOp?
model/conv2d/BiasAdd_1BiasAddmodel/conv2d/Conv2D_1:output:0-model/conv2d/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
model/conv2d/BiasAdd_1?
model/conv2d/Relu_1Relumodel/conv2d/BiasAdd_1:output:0*
T0*1
_output_shapes
:??????????? 2
model/conv2d/Relu_1?
model/max_pooling2d/MaxPool_1MaxPool!model/conv2d/Relu_1:activations:0*0
_output_shapes
:??????????n *
ksize
*
paddingVALID*
strides
2
model/max_pooling2d/MaxPool_1?
model/dropout/Identity_1Identity&model/max_pooling2d/MaxPool_1:output:0*
T0*0
_output_shapes
:??????????n 2
model/dropout/Identity_1?
&model/conv2d_1/Conv2D_1/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02(
&model/conv2d_1/Conv2D_1/ReadVariableOp?
model/conv2d_1/Conv2D_1Conv2D!model/dropout/Identity_1:output:0.model/conv2d_1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????n@*
paddingSAME*
strides
2
model/conv2d_1/Conv2D_1?
'model/conv2d_1/BiasAdd_1/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model/conv2d_1/BiasAdd_1/ReadVariableOp?
model/conv2d_1/BiasAdd_1BiasAdd model/conv2d_1/Conv2D_1:output:0/model/conv2d_1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????n@2
model/conv2d_1/BiasAdd_1?
model/conv2d_1/Relu_1Relu!model/conv2d_1/BiasAdd_1:output:0*
T0*0
_output_shapes
:??????????n@2
model/conv2d_1/Relu_1?
model/max_pooling2d_1/MaxPool_1MaxPool#model/conv2d_1/Relu_1:activations:0*/
_output_shapes
:?????????n7@*
ksize
*
paddingVALID*
strides
2!
model/max_pooling2d_1/MaxPool_1?
&model/conv2d_2/Conv2D_1/ReadVariableOpReadVariableOp-model_conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02(
&model/conv2d_2/Conv2D_1/ReadVariableOp?
model/conv2d_2/Conv2D_1Conv2D(model/max_pooling2d_1/MaxPool_1:output:0.model/conv2d_2/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????n7?*
paddingSAME*
strides
2
model/conv2d_2/Conv2D_1?
'model/conv2d_2/BiasAdd_1/ReadVariableOpReadVariableOp.model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'model/conv2d_2/BiasAdd_1/ReadVariableOp?
model/conv2d_2/BiasAdd_1BiasAdd model/conv2d_2/Conv2D_1:output:0/model/conv2d_2/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????n7?2
model/conv2d_2/BiasAdd_1?
model/conv2d_2/Relu_1Relu!model/conv2d_2/BiasAdd_1:output:0*
T0*0
_output_shapes
:?????????n7?2
model/conv2d_2/Relu_1?
model/max_pooling2d_2/MaxPool_1MaxPool#model/conv2d_2/Relu_1:activations:0*0
_output_shapes
:?????????7?*
ksize
*
paddingVALID*
strides
2!
model/max_pooling2d_2/MaxPool_1?
&model/conv2d_3/Conv2D_1/ReadVariableOpReadVariableOp-model_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&model/conv2d_3/Conv2D_1/ReadVariableOp?
model/conv2d_3/Conv2D_1Conv2D(model/max_pooling2d_2/MaxPool_1:output:0.model/conv2d_3/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????7?*
paddingSAME*
strides
2
model/conv2d_3/Conv2D_1?
'model/conv2d_3/BiasAdd_1/ReadVariableOpReadVariableOp.model_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'model/conv2d_3/BiasAdd_1/ReadVariableOp?
model/conv2d_3/BiasAdd_1BiasAdd model/conv2d_3/Conv2D_1:output:0/model/conv2d_3/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????7?2
model/conv2d_3/BiasAdd_1?
model/conv2d_3/Relu_1Relu!model/conv2d_3/BiasAdd_1:output:0*
T0*0
_output_shapes
:?????????7?2
model/conv2d_3/Relu_1?
model/max_pooling2d_3/MaxPool_1MaxPool#model/conv2d_3/Relu_1:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2!
model/max_pooling2d_3/MaxPool_1?
&model/conv2d_4/Conv2D_1/ReadVariableOpReadVariableOp-model_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&model/conv2d_4/Conv2D_1/ReadVariableOp?
model/conv2d_4/Conv2D_1Conv2D(model/max_pooling2d_3/MaxPool_1:output:0.model/conv2d_4/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model/conv2d_4/Conv2D_1?
'model/conv2d_4/BiasAdd_1/ReadVariableOpReadVariableOp.model_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'model/conv2d_4/BiasAdd_1/ReadVariableOp?
model/conv2d_4/BiasAdd_1BiasAdd model/conv2d_4/Conv2D_1:output:0/model/conv2d_4/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model/conv2d_4/BiasAdd_1?
model/conv2d_4/Relu_1Relu!model/conv2d_4/BiasAdd_1:output:0*
T0*0
_output_shapes
:??????????2
model/conv2d_4/Relu_1?
model/dropout_1/Identity_1Identity#model/conv2d_4/Relu_1:activations:0*
T0*0
_output_shapes
:??????????2
model/dropout_1/Identity_1?
7model/global_average_pooling2d/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      29
7model/global_average_pooling2d/Mean_1/reduction_indices?
%model/global_average_pooling2d/Mean_1Mean#model/dropout_1/Identity_1:output:0@model/global_average_pooling2d/Mean_1/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2'
%model/global_average_pooling2d/Mean_1?
#model/dense/MatMul_1/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02%
#model/dense/MatMul_1/ReadVariableOp?
model/dense/MatMul_1MatMul.model/global_average_pooling2d/Mean_1:output:0+model/dense/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense/MatMul_1?
$model/dense/BiasAdd_1/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense/BiasAdd_1/ReadVariableOp?
model/dense/BiasAdd_1BiasAddmodel/dense/MatMul_1:product:0,model/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense/BiasAdd_1?
tf.math.subtract/SubSubmodel/dense/BiasAdd:output:0model/dense/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract/Sub?
tf.math.square/SquareSquaretf.math.subtract/Sub:z:0*
T0*'
_output_shapes
:?????????2
tf.math.square/Square?
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2*
(tf.math.reduce_sum/Sum/reduction_indices?
tf.math.reduce_sum/SumSumtf.math.square/Square:y:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
tf.math.reduce_sum/Sum?
&tf.clip_by_value/clip_by_value/MinimumMinimumtf.math.reduce_sum/Sum:output:0(tf_clip_by_value_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????2(
&tf.clip_by_value/clip_by_value/Minimum?
tf.clip_by_value/clip_by_valueMaximum*tf.clip_by_value/clip_by_value/Minimum:z:0 tf_clip_by_value_clip_by_value_y*
T0*'
_output_shapes
:?????????2 
tf.clip_by_value/clip_by_value?
tf.math.sqrt/SqrtSqrt"tf.clip_by_value/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
tf.math.sqrt/Sqrt?
IdentityIdentitytf.math.sqrt/Sqrt:y:0$^model/conv2d/BiasAdd/ReadVariableOp&^model/conv2d/BiasAdd_1/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp%^model/conv2d/Conv2D_1/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp(^model/conv2d_1/BiasAdd_1/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp'^model/conv2d_1/Conv2D_1/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp(^model/conv2d_2/BiasAdd_1/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp'^model/conv2d_2/Conv2D_1/ReadVariableOp&^model/conv2d_3/BiasAdd/ReadVariableOp(^model/conv2d_3/BiasAdd_1/ReadVariableOp%^model/conv2d_3/Conv2D/ReadVariableOp'^model/conv2d_3/Conv2D_1/ReadVariableOp&^model/conv2d_4/BiasAdd/ReadVariableOp(^model/conv2d_4/BiasAdd_1/ReadVariableOp%^model/conv2d_4/Conv2D/ReadVariableOp'^model/conv2d_4/Conv2D_1/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp%^model/dense/BiasAdd_1/ReadVariableOp"^model/dense/MatMul/ReadVariableOp$^model/dense/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesp
n:???????????:???????????::::::::::::: : 2J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2N
%model/conv2d/BiasAdd_1/ReadVariableOp%model/conv2d/BiasAdd_1/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2L
$model/conv2d/Conv2D_1/ReadVariableOp$model/conv2d/Conv2D_1/ReadVariableOp2N
%model/conv2d_1/BiasAdd/ReadVariableOp%model/conv2d_1/BiasAdd/ReadVariableOp2R
'model/conv2d_1/BiasAdd_1/ReadVariableOp'model/conv2d_1/BiasAdd_1/ReadVariableOp2L
$model/conv2d_1/Conv2D/ReadVariableOp$model/conv2d_1/Conv2D/ReadVariableOp2P
&model/conv2d_1/Conv2D_1/ReadVariableOp&model/conv2d_1/Conv2D_1/ReadVariableOp2N
%model/conv2d_2/BiasAdd/ReadVariableOp%model/conv2d_2/BiasAdd/ReadVariableOp2R
'model/conv2d_2/BiasAdd_1/ReadVariableOp'model/conv2d_2/BiasAdd_1/ReadVariableOp2L
$model/conv2d_2/Conv2D/ReadVariableOp$model/conv2d_2/Conv2D/ReadVariableOp2P
&model/conv2d_2/Conv2D_1/ReadVariableOp&model/conv2d_2/Conv2D_1/ReadVariableOp2N
%model/conv2d_3/BiasAdd/ReadVariableOp%model/conv2d_3/BiasAdd/ReadVariableOp2R
'model/conv2d_3/BiasAdd_1/ReadVariableOp'model/conv2d_3/BiasAdd_1/ReadVariableOp2L
$model/conv2d_3/Conv2D/ReadVariableOp$model/conv2d_3/Conv2D/ReadVariableOp2P
&model/conv2d_3/Conv2D_1/ReadVariableOp&model/conv2d_3/Conv2D_1/ReadVariableOp2N
%model/conv2d_4/BiasAdd/ReadVariableOp%model/conv2d_4/BiasAdd/ReadVariableOp2R
'model/conv2d_4/BiasAdd_1/ReadVariableOp'model/conv2d_4/BiasAdd_1/ReadVariableOp2L
$model/conv2d_4/Conv2D/ReadVariableOp$model/conv2d_4/Conv2D/ReadVariableOp2P
&model/conv2d_4/Conv2D_1/ReadVariableOp&model/conv2d_4/Conv2D_1/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2L
$model/dense/BiasAdd_1/ReadVariableOp$model/dense/BiasAdd_1/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2J
#model/dense/MatMul_1/ReadVariableOp#model/dense/MatMul_1/ReadVariableOp:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: 
?

?
B__inference_conv2d_3_layer_call_and_return_conditional_losses_8737

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????7?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????7?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????7?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????7?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????7?::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????7?
 
_user_specified_nameinputs
?9
?
?__inference_model_layer_call_and_return_conditional_losses_7603
input_3
conv2d_7398
conv2d_7400
conv2d_1_7456
conv2d_1_7458
conv2d_2_7484
conv2d_2_7486
conv2d_3_7512
conv2d_3_7514
conv2d_4_7540
conv2d_4_7542

dense_7597

dense_7599
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?dense/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_7398conv2d_7400*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_73872 
conv2d/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????n * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_73172
max_pooling2d/PartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????n * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_74162!
dropout/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv2d_1_7456conv2d_1_7458*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????n@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_74452"
 conv2d_1/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????n7@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_73292!
max_pooling2d_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_7484conv2d_2_7486*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????n7?*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_74732"
 conv2d_2/StatefulPartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????7?* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_73412!
max_pooling2d_2/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_3_7512conv2d_3_7514*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????7?*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_75012"
 conv2d_3/StatefulPartitionedCall?
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_73532!
max_pooling2d_3/PartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_4_7540conv2d_4_7542*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_75292"
 conv2d_4/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_75572#
!dropout_1/StatefulPartitionedCall?
(global_average_pooling2d/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *[
fVRT
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_73662*
(global_average_pooling2d/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0
dense_7597
dense_7599*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_75862
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:???????????::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_3
?
n
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_7366

inputs
identity?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
y
$__inference_dense_layer_call_fn_8812

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_75862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
?__inference_dense_layer_call_and_return_conditional_losses_7586

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
H
,__inference_max_pooling2d_layer_call_fn_7323

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_73172
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
B__inference_conv2d_4_layer_call_and_return_conditional_losses_8757

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
$__inference_model_layer_call_fn_7715
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_76882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:???????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_3
?
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_7317

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
@__inference_conv2d_layer_call_and_return_conditional_losses_8650

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
|
'__inference_conv2d_4_layer_call_fn_8766

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_75292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
C__inference_dropout_1_layer_call_and_return_conditional_losses_7557

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_7329

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_7562

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?6
?
?__inference_model_layer_call_and_return_conditional_losses_7758

inputs
conv2d_7720
conv2d_7722
conv2d_1_7727
conv2d_1_7729
conv2d_2_7733
conv2d_2_7735
conv2d_3_7739
conv2d_3_7741
conv2d_4_7745
conv2d_4_7747

dense_7752

dense_7754
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_7720conv2d_7722*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_73872 
conv2d/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????n * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_73172
max_pooling2d/PartitionedCall?
dropout/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????n * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_74212
dropout/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv2d_1_7727conv2d_1_7729*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????n@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_74452"
 conv2d_1/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????n7@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_73292!
max_pooling2d_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_7733conv2d_2_7735*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????n7?*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_74732"
 conv2d_2/StatefulPartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????7?* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_73412!
max_pooling2d_2/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_3_7739conv2d_3_7741*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????7?*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_75012"
 conv2d_3/StatefulPartitionedCall?
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_73532!
max_pooling2d_3/PartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_4_7745conv2d_4_7747*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_75292"
 conv2d_4/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_75622
dropout_1/PartitionedCall?
(global_average_pooling2d/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *[
fVRT
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_73662*
(global_average_pooling2d/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0
dense_7752
dense_7754*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_75862
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:???????????::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
|
'__inference_conv2d_3_layer_call_fn_8746

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????7?*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_75012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????7?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????7?::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????7?
 
_user_specified_nameinputs
?
D
(__inference_dropout_1_layer_call_fn_8793

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_75622
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?9
?
?__inference_model_layer_call_and_return_conditional_losses_7688

inputs
conv2d_7650
conv2d_7652
conv2d_1_7657
conv2d_1_7659
conv2d_2_7663
conv2d_2_7665
conv2d_3_7669
conv2d_3_7671
conv2d_4_7675
conv2d_4_7677

dense_7682

dense_7684
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?dense/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_7650conv2d_7652*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_73872 
conv2d/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????n * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_73172
max_pooling2d/PartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????n * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_74162!
dropout/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv2d_1_7657conv2d_1_7659*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????n@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_74452"
 conv2d_1/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????n7@* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_73292!
max_pooling2d_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_7663conv2d_2_7665*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????n7?*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_74732"
 conv2d_2/StatefulPartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????7?* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_73412!
max_pooling2d_2/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_3_7669conv2d_3_7671*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????7?*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_75012"
 conv2d_3/StatefulPartitionedCall?
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_73532!
max_pooling2d_3/PartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv2d_4_7675conv2d_4_7677*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_75292"
 conv2d_4/StatefulPartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_75572#
!dropout_1/StatefulPartitionedCall?
(global_average_pooling2d/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *[
fVRT
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_73662*
(global_average_pooling2d/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0
dense_7682
dense_7684*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_75862
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:???????????::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
 __inference__traced_restore_9106
file_prefix
assignvariableop_adam_iter"
assignvariableop_1_adam_beta_1"
assignvariableop_2_adam_beta_2!
assignvariableop_3_adam_decay)
%assignvariableop_4_adam_learning_rate$
 assignvariableop_5_conv2d_kernel"
assignvariableop_6_conv2d_bias&
"assignvariableop_7_conv2d_1_kernel$
 assignvariableop_8_conv2d_1_bias&
"assignvariableop_9_conv2d_2_kernel%
!assignvariableop_10_conv2d_2_bias'
#assignvariableop_11_conv2d_3_kernel%
!assignvariableop_12_conv2d_3_bias'
#assignvariableop_13_conv2d_4_kernel%
!assignvariableop_14_conv2d_4_bias$
 assignvariableop_15_dense_kernel"
assignvariableop_16_dense_bias
assignvariableop_17_total
assignvariableop_18_count,
(assignvariableop_19_adam_conv2d_kernel_m*
&assignvariableop_20_adam_conv2d_bias_m.
*assignvariableop_21_adam_conv2d_1_kernel_m,
(assignvariableop_22_adam_conv2d_1_bias_m.
*assignvariableop_23_adam_conv2d_2_kernel_m,
(assignvariableop_24_adam_conv2d_2_bias_m.
*assignvariableop_25_adam_conv2d_3_kernel_m,
(assignvariableop_26_adam_conv2d_3_bias_m.
*assignvariableop_27_adam_conv2d_4_kernel_m,
(assignvariableop_28_adam_conv2d_4_bias_m+
'assignvariableop_29_adam_dense_kernel_m)
%assignvariableop_30_adam_dense_bias_m,
(assignvariableop_31_adam_conv2d_kernel_v*
&assignvariableop_32_adam_conv2d_bias_v.
*assignvariableop_33_adam_conv2d_1_kernel_v,
(assignvariableop_34_adam_conv2d_1_bias_v.
*assignvariableop_35_adam_conv2d_2_kernel_v,
(assignvariableop_36_adam_conv2d_2_bias_v.
*assignvariableop_37_adam_conv2d_3_kernel_v,
(assignvariableop_38_adam_conv2d_3_bias_v.
*assignvariableop_39_adam_conv2d_4_kernel_v,
(assignvariableop_40_adam_conv2d_4_bias_v+
'assignvariableop_41_adam_dense_kernel_v)
%assignvariableop_42_adam_dense_bias_v
identity_44??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*?
value?B?,B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::*:
dtypes0
.2,	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_conv2d_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp"assignvariableop_7_conv2d_1_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp assignvariableop_8_conv2d_1_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp"assignvariableop_9_conv2d_2_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp!assignvariableop_10_conv2d_2_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp#assignvariableop_11_conv2d_3_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp!assignvariableop_12_conv2d_3_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp#assignvariableop_13_conv2d_4_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp!assignvariableop_14_conv2d_4_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_dense_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_conv2d_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_conv2d_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_conv2d_1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_conv2d_1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_conv2d_2_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_conv2d_2_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv2d_3_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv2d_3_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_conv2d_4_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv2d_4_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_dense_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp%assignvariableop_30_adam_dense_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_conv2d_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp&assignvariableop_32_adam_conv2d_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv2d_1_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_conv2d_1_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_conv2d_2_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_conv2d_2_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv2d_3_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv2d_3_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_conv2d_4_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_conv2d_4_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp'assignvariableop_41_adam_dense_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp%assignvariableop_42_adam_dense_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_429
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_43Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_43?
Identity_44IdentityIdentity_43:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_44"#
identity_44Identity_44:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?	
?
?__inference_dense_layer_call_and_return_conditional_losses_8803

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
J
.__inference_max_pooling2d_3_layer_call_fn_7359

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_73532
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
2__inference_sig_ver_siamese_cnn_layer_call_fn_8461
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *V
fQRO
M__inference_sig_ver_siamese_cnn_layer_call_and_return_conditional_losses_80902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesp
n:???????????:???????????::::::::::::: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: 
?
B
&__inference_dropout_layer_call_fn_8686

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????n * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_74212
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????n 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????n :X T
0
_output_shapes
:??????????n 
 
_user_specified_nameinputs
?
S
7__inference_global_average_pooling2d_layer_call_fn_7372

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *[
fVRT
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_73662
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_8783

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
$__inference_model_layer_call_fn_8639

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_77582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:???????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_7341

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
_
A__inference_dropout_layer_call_and_return_conditional_losses_7421

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????n 2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????n 2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:??????????n :X T
0
_output_shapes
:??????????n 
 
_user_specified_nameinputs
?
?
M__inference_sig_ver_siamese_cnn_layer_call_and_return_conditional_losses_7896
input_1
input_2

model_7848

model_7850

model_7852

model_7854

model_7856

model_7858

model_7860

model_7862

model_7864

model_7866

model_7868

model_7870,
(tf_clip_by_value_clip_by_value_minimum_y$
 tf_clip_by_value_clip_by_value_y
identity??model/StatefulPartitionedCall?model/StatefulPartitionedCall_1?
model/StatefulPartitionedCallStatefulPartitionedCallinput_1
model_7848
model_7850
model_7852
model_7854
model_7856
model_7858
model_7860
model_7862
model_7864
model_7866
model_7868
model_7870*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_76882
model/StatefulPartitionedCall?
model/StatefulPartitionedCall_1StatefulPartitionedCallinput_2
model_7848
model_7850
model_7852
model_7854
model_7856
model_7858
model_7860
model_7862
model_7864
model_7866
model_7868
model_7870*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_76882!
model/StatefulPartitionedCall_1?
tf.math.subtract/SubSub&model/StatefulPartitionedCall:output:0(model/StatefulPartitionedCall_1:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract/Sub?
tf.math.square/SquareSquaretf.math.subtract/Sub:z:0*
T0*'
_output_shapes
:?????????2
tf.math.square/Square?
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2*
(tf.math.reduce_sum/Sum/reduction_indices?
tf.math.reduce_sum/SumSumtf.math.square/Square:y:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
tf.math.reduce_sum/Sum?
&tf.clip_by_value/clip_by_value/MinimumMinimumtf.math.reduce_sum/Sum:output:0(tf_clip_by_value_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????2(
&tf.clip_by_value/clip_by_value/Minimum?
tf.clip_by_value/clip_by_valueMaximum*tf.clip_by_value/clip_by_value/Minimum:z:0 tf_clip_by_value_clip_by_value_y*
T0*'
_output_shapes
:?????????2 
tf.clip_by_value/clip_by_value?
tf.math.sqrt/SqrtSqrt"tf.clip_by_value/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
tf.math.sqrt/Sqrt?
IdentityIdentitytf.math.sqrt/Sqrt:y:0^model/StatefulPartitionedCall ^model/StatefulPartitionedCall_1*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesp
n:???????????:???????????::::::::::::: : 2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model/StatefulPartitionedCall_1model/StatefulPartitionedCall_1:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1:ZV
1
_output_shapes
:???????????
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: 
?

?
B__inference_conv2d_3_layer_call_and_return_conditional_losses_7501

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????7?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????7?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????7?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????7?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????7?::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????7?
 
_user_specified_nameinputs
?
_
&__inference_dropout_layer_call_fn_8681

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????n * 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_74162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????n 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????n 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????n 
 
_user_specified_nameinputs
?
?
2__inference_sig_ver_siamese_cnn_layer_call_fn_8427
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *V
fQRO
M__inference_sig_ver_siamese_cnn_layer_call_and_return_conditional_losses_80042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesp
n:???????????:???????????::::::::::::: : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: 
?

?
@__inference_conv2d_layer_call_and_return_conditional_losses_7387

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
__inference__wrapped_model_7311
input_1
input_2C
?sig_ver_siamese_cnn_model_conv2d_conv2d_readvariableop_resourceD
@sig_ver_siamese_cnn_model_conv2d_biasadd_readvariableop_resourceE
Asig_ver_siamese_cnn_model_conv2d_1_conv2d_readvariableop_resourceF
Bsig_ver_siamese_cnn_model_conv2d_1_biasadd_readvariableop_resourceE
Asig_ver_siamese_cnn_model_conv2d_2_conv2d_readvariableop_resourceF
Bsig_ver_siamese_cnn_model_conv2d_2_biasadd_readvariableop_resourceE
Asig_ver_siamese_cnn_model_conv2d_3_conv2d_readvariableop_resourceF
Bsig_ver_siamese_cnn_model_conv2d_3_biasadd_readvariableop_resourceE
Asig_ver_siamese_cnn_model_conv2d_4_conv2d_readvariableop_resourceF
Bsig_ver_siamese_cnn_model_conv2d_4_biasadd_readvariableop_resourceB
>sig_ver_siamese_cnn_model_dense_matmul_readvariableop_resourceC
?sig_ver_siamese_cnn_model_dense_biasadd_readvariableop_resource@
<sig_ver_siamese_cnn_tf_clip_by_value_clip_by_value_minimum_y8
4sig_ver_siamese_cnn_tf_clip_by_value_clip_by_value_y
identity??7sig_ver_siamese_cnn/model/conv2d/BiasAdd/ReadVariableOp?9sig_ver_siamese_cnn/model/conv2d/BiasAdd_1/ReadVariableOp?6sig_ver_siamese_cnn/model/conv2d/Conv2D/ReadVariableOp?8sig_ver_siamese_cnn/model/conv2d/Conv2D_1/ReadVariableOp?9sig_ver_siamese_cnn/model/conv2d_1/BiasAdd/ReadVariableOp?;sig_ver_siamese_cnn/model/conv2d_1/BiasAdd_1/ReadVariableOp?8sig_ver_siamese_cnn/model/conv2d_1/Conv2D/ReadVariableOp?:sig_ver_siamese_cnn/model/conv2d_1/Conv2D_1/ReadVariableOp?9sig_ver_siamese_cnn/model/conv2d_2/BiasAdd/ReadVariableOp?;sig_ver_siamese_cnn/model/conv2d_2/BiasAdd_1/ReadVariableOp?8sig_ver_siamese_cnn/model/conv2d_2/Conv2D/ReadVariableOp?:sig_ver_siamese_cnn/model/conv2d_2/Conv2D_1/ReadVariableOp?9sig_ver_siamese_cnn/model/conv2d_3/BiasAdd/ReadVariableOp?;sig_ver_siamese_cnn/model/conv2d_3/BiasAdd_1/ReadVariableOp?8sig_ver_siamese_cnn/model/conv2d_3/Conv2D/ReadVariableOp?:sig_ver_siamese_cnn/model/conv2d_3/Conv2D_1/ReadVariableOp?9sig_ver_siamese_cnn/model/conv2d_4/BiasAdd/ReadVariableOp?;sig_ver_siamese_cnn/model/conv2d_4/BiasAdd_1/ReadVariableOp?8sig_ver_siamese_cnn/model/conv2d_4/Conv2D/ReadVariableOp?:sig_ver_siamese_cnn/model/conv2d_4/Conv2D_1/ReadVariableOp?6sig_ver_siamese_cnn/model/dense/BiasAdd/ReadVariableOp?8sig_ver_siamese_cnn/model/dense/BiasAdd_1/ReadVariableOp?5sig_ver_siamese_cnn/model/dense/MatMul/ReadVariableOp?7sig_ver_siamese_cnn/model/dense/MatMul_1/ReadVariableOp?
6sig_ver_siamese_cnn/model/conv2d/Conv2D/ReadVariableOpReadVariableOp?sig_ver_siamese_cnn_model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype028
6sig_ver_siamese_cnn/model/conv2d/Conv2D/ReadVariableOp?
'sig_ver_siamese_cnn/model/conv2d/Conv2DConv2Dinput_1>sig_ver_siamese_cnn/model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2)
'sig_ver_siamese_cnn/model/conv2d/Conv2D?
7sig_ver_siamese_cnn/model/conv2d/BiasAdd/ReadVariableOpReadVariableOp@sig_ver_siamese_cnn_model_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype029
7sig_ver_siamese_cnn/model/conv2d/BiasAdd/ReadVariableOp?
(sig_ver_siamese_cnn/model/conv2d/BiasAddBiasAdd0sig_ver_siamese_cnn/model/conv2d/Conv2D:output:0?sig_ver_siamese_cnn/model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2*
(sig_ver_siamese_cnn/model/conv2d/BiasAdd?
%sig_ver_siamese_cnn/model/conv2d/ReluRelu1sig_ver_siamese_cnn/model/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2'
%sig_ver_siamese_cnn/model/conv2d/Relu?
/sig_ver_siamese_cnn/model/max_pooling2d/MaxPoolMaxPool3sig_ver_siamese_cnn/model/conv2d/Relu:activations:0*0
_output_shapes
:??????????n *
ksize
*
paddingVALID*
strides
21
/sig_ver_siamese_cnn/model/max_pooling2d/MaxPool?
*sig_ver_siamese_cnn/model/dropout/IdentityIdentity8sig_ver_siamese_cnn/model/max_pooling2d/MaxPool:output:0*
T0*0
_output_shapes
:??????????n 2,
*sig_ver_siamese_cnn/model/dropout/Identity?
8sig_ver_siamese_cnn/model/conv2d_1/Conv2D/ReadVariableOpReadVariableOpAsig_ver_siamese_cnn_model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02:
8sig_ver_siamese_cnn/model/conv2d_1/Conv2D/ReadVariableOp?
)sig_ver_siamese_cnn/model/conv2d_1/Conv2DConv2D3sig_ver_siamese_cnn/model/dropout/Identity:output:0@sig_ver_siamese_cnn/model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????n@*
paddingSAME*
strides
2+
)sig_ver_siamese_cnn/model/conv2d_1/Conv2D?
9sig_ver_siamese_cnn/model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpBsig_ver_siamese_cnn_model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02;
9sig_ver_siamese_cnn/model/conv2d_1/BiasAdd/ReadVariableOp?
*sig_ver_siamese_cnn/model/conv2d_1/BiasAddBiasAdd2sig_ver_siamese_cnn/model/conv2d_1/Conv2D:output:0Asig_ver_siamese_cnn/model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????n@2,
*sig_ver_siamese_cnn/model/conv2d_1/BiasAdd?
'sig_ver_siamese_cnn/model/conv2d_1/ReluRelu3sig_ver_siamese_cnn/model/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????n@2)
'sig_ver_siamese_cnn/model/conv2d_1/Relu?
1sig_ver_siamese_cnn/model/max_pooling2d_1/MaxPoolMaxPool5sig_ver_siamese_cnn/model/conv2d_1/Relu:activations:0*/
_output_shapes
:?????????n7@*
ksize
*
paddingVALID*
strides
23
1sig_ver_siamese_cnn/model/max_pooling2d_1/MaxPool?
8sig_ver_siamese_cnn/model/conv2d_2/Conv2D/ReadVariableOpReadVariableOpAsig_ver_siamese_cnn_model_conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02:
8sig_ver_siamese_cnn/model/conv2d_2/Conv2D/ReadVariableOp?
)sig_ver_siamese_cnn/model/conv2d_2/Conv2DConv2D:sig_ver_siamese_cnn/model/max_pooling2d_1/MaxPool:output:0@sig_ver_siamese_cnn/model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????n7?*
paddingSAME*
strides
2+
)sig_ver_siamese_cnn/model/conv2d_2/Conv2D?
9sig_ver_siamese_cnn/model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpBsig_ver_siamese_cnn_model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02;
9sig_ver_siamese_cnn/model/conv2d_2/BiasAdd/ReadVariableOp?
*sig_ver_siamese_cnn/model/conv2d_2/BiasAddBiasAdd2sig_ver_siamese_cnn/model/conv2d_2/Conv2D:output:0Asig_ver_siamese_cnn/model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????n7?2,
*sig_ver_siamese_cnn/model/conv2d_2/BiasAdd?
'sig_ver_siamese_cnn/model/conv2d_2/ReluRelu3sig_ver_siamese_cnn/model/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????n7?2)
'sig_ver_siamese_cnn/model/conv2d_2/Relu?
1sig_ver_siamese_cnn/model/max_pooling2d_2/MaxPoolMaxPool5sig_ver_siamese_cnn/model/conv2d_2/Relu:activations:0*0
_output_shapes
:?????????7?*
ksize
*
paddingVALID*
strides
23
1sig_ver_siamese_cnn/model/max_pooling2d_2/MaxPool?
8sig_ver_siamese_cnn/model/conv2d_3/Conv2D/ReadVariableOpReadVariableOpAsig_ver_siamese_cnn_model_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02:
8sig_ver_siamese_cnn/model/conv2d_3/Conv2D/ReadVariableOp?
)sig_ver_siamese_cnn/model/conv2d_3/Conv2DConv2D:sig_ver_siamese_cnn/model/max_pooling2d_2/MaxPool:output:0@sig_ver_siamese_cnn/model/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????7?*
paddingSAME*
strides
2+
)sig_ver_siamese_cnn/model/conv2d_3/Conv2D?
9sig_ver_siamese_cnn/model/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpBsig_ver_siamese_cnn_model_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02;
9sig_ver_siamese_cnn/model/conv2d_3/BiasAdd/ReadVariableOp?
*sig_ver_siamese_cnn/model/conv2d_3/BiasAddBiasAdd2sig_ver_siamese_cnn/model/conv2d_3/Conv2D:output:0Asig_ver_siamese_cnn/model/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????7?2,
*sig_ver_siamese_cnn/model/conv2d_3/BiasAdd?
'sig_ver_siamese_cnn/model/conv2d_3/ReluRelu3sig_ver_siamese_cnn/model/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????7?2)
'sig_ver_siamese_cnn/model/conv2d_3/Relu?
1sig_ver_siamese_cnn/model/max_pooling2d_3/MaxPoolMaxPool5sig_ver_siamese_cnn/model/conv2d_3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
23
1sig_ver_siamese_cnn/model/max_pooling2d_3/MaxPool?
8sig_ver_siamese_cnn/model/conv2d_4/Conv2D/ReadVariableOpReadVariableOpAsig_ver_siamese_cnn_model_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02:
8sig_ver_siamese_cnn/model/conv2d_4/Conv2D/ReadVariableOp?
)sig_ver_siamese_cnn/model/conv2d_4/Conv2DConv2D:sig_ver_siamese_cnn/model/max_pooling2d_3/MaxPool:output:0@sig_ver_siamese_cnn/model/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2+
)sig_ver_siamese_cnn/model/conv2d_4/Conv2D?
9sig_ver_siamese_cnn/model/conv2d_4/BiasAdd/ReadVariableOpReadVariableOpBsig_ver_siamese_cnn_model_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02;
9sig_ver_siamese_cnn/model/conv2d_4/BiasAdd/ReadVariableOp?
*sig_ver_siamese_cnn/model/conv2d_4/BiasAddBiasAdd2sig_ver_siamese_cnn/model/conv2d_4/Conv2D:output:0Asig_ver_siamese_cnn/model/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2,
*sig_ver_siamese_cnn/model/conv2d_4/BiasAdd?
'sig_ver_siamese_cnn/model/conv2d_4/ReluRelu3sig_ver_siamese_cnn/model/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2)
'sig_ver_siamese_cnn/model/conv2d_4/Relu?
,sig_ver_siamese_cnn/model/dropout_1/IdentityIdentity5sig_ver_siamese_cnn/model/conv2d_4/Relu:activations:0*
T0*0
_output_shapes
:??????????2.
,sig_ver_siamese_cnn/model/dropout_1/Identity?
Isig_ver_siamese_cnn/model/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2K
Isig_ver_siamese_cnn/model/global_average_pooling2d/Mean/reduction_indices?
7sig_ver_siamese_cnn/model/global_average_pooling2d/MeanMean5sig_ver_siamese_cnn/model/dropout_1/Identity:output:0Rsig_ver_siamese_cnn/model/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????29
7sig_ver_siamese_cnn/model/global_average_pooling2d/Mean?
5sig_ver_siamese_cnn/model/dense/MatMul/ReadVariableOpReadVariableOp>sig_ver_siamese_cnn_model_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype027
5sig_ver_siamese_cnn/model/dense/MatMul/ReadVariableOp?
&sig_ver_siamese_cnn/model/dense/MatMulMatMul@sig_ver_siamese_cnn/model/global_average_pooling2d/Mean:output:0=sig_ver_siamese_cnn/model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2(
&sig_ver_siamese_cnn/model/dense/MatMul?
6sig_ver_siamese_cnn/model/dense/BiasAdd/ReadVariableOpReadVariableOp?sig_ver_siamese_cnn_model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6sig_ver_siamese_cnn/model/dense/BiasAdd/ReadVariableOp?
'sig_ver_siamese_cnn/model/dense/BiasAddBiasAdd0sig_ver_siamese_cnn/model/dense/MatMul:product:0>sig_ver_siamese_cnn/model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'sig_ver_siamese_cnn/model/dense/BiasAdd?
8sig_ver_siamese_cnn/model/conv2d/Conv2D_1/ReadVariableOpReadVariableOp?sig_ver_siamese_cnn_model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02:
8sig_ver_siamese_cnn/model/conv2d/Conv2D_1/ReadVariableOp?
)sig_ver_siamese_cnn/model/conv2d/Conv2D_1Conv2Dinput_2@sig_ver_siamese_cnn/model/conv2d/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2+
)sig_ver_siamese_cnn/model/conv2d/Conv2D_1?
9sig_ver_siamese_cnn/model/conv2d/BiasAdd_1/ReadVariableOpReadVariableOp@sig_ver_siamese_cnn_model_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02;
9sig_ver_siamese_cnn/model/conv2d/BiasAdd_1/ReadVariableOp?
*sig_ver_siamese_cnn/model/conv2d/BiasAdd_1BiasAdd2sig_ver_siamese_cnn/model/conv2d/Conv2D_1:output:0Asig_ver_siamese_cnn/model/conv2d/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2,
*sig_ver_siamese_cnn/model/conv2d/BiasAdd_1?
'sig_ver_siamese_cnn/model/conv2d/Relu_1Relu3sig_ver_siamese_cnn/model/conv2d/BiasAdd_1:output:0*
T0*1
_output_shapes
:??????????? 2)
'sig_ver_siamese_cnn/model/conv2d/Relu_1?
1sig_ver_siamese_cnn/model/max_pooling2d/MaxPool_1MaxPool5sig_ver_siamese_cnn/model/conv2d/Relu_1:activations:0*0
_output_shapes
:??????????n *
ksize
*
paddingVALID*
strides
23
1sig_ver_siamese_cnn/model/max_pooling2d/MaxPool_1?
,sig_ver_siamese_cnn/model/dropout/Identity_1Identity:sig_ver_siamese_cnn/model/max_pooling2d/MaxPool_1:output:0*
T0*0
_output_shapes
:??????????n 2.
,sig_ver_siamese_cnn/model/dropout/Identity_1?
:sig_ver_siamese_cnn/model/conv2d_1/Conv2D_1/ReadVariableOpReadVariableOpAsig_ver_siamese_cnn_model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02<
:sig_ver_siamese_cnn/model/conv2d_1/Conv2D_1/ReadVariableOp?
+sig_ver_siamese_cnn/model/conv2d_1/Conv2D_1Conv2D5sig_ver_siamese_cnn/model/dropout/Identity_1:output:0Bsig_ver_siamese_cnn/model/conv2d_1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????n@*
paddingSAME*
strides
2-
+sig_ver_siamese_cnn/model/conv2d_1/Conv2D_1?
;sig_ver_siamese_cnn/model/conv2d_1/BiasAdd_1/ReadVariableOpReadVariableOpBsig_ver_siamese_cnn_model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;sig_ver_siamese_cnn/model/conv2d_1/BiasAdd_1/ReadVariableOp?
,sig_ver_siamese_cnn/model/conv2d_1/BiasAdd_1BiasAdd4sig_ver_siamese_cnn/model/conv2d_1/Conv2D_1:output:0Csig_ver_siamese_cnn/model/conv2d_1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????n@2.
,sig_ver_siamese_cnn/model/conv2d_1/BiasAdd_1?
)sig_ver_siamese_cnn/model/conv2d_1/Relu_1Relu5sig_ver_siamese_cnn/model/conv2d_1/BiasAdd_1:output:0*
T0*0
_output_shapes
:??????????n@2+
)sig_ver_siamese_cnn/model/conv2d_1/Relu_1?
3sig_ver_siamese_cnn/model/max_pooling2d_1/MaxPool_1MaxPool7sig_ver_siamese_cnn/model/conv2d_1/Relu_1:activations:0*/
_output_shapes
:?????????n7@*
ksize
*
paddingVALID*
strides
25
3sig_ver_siamese_cnn/model/max_pooling2d_1/MaxPool_1?
:sig_ver_siamese_cnn/model/conv2d_2/Conv2D_1/ReadVariableOpReadVariableOpAsig_ver_siamese_cnn_model_conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02<
:sig_ver_siamese_cnn/model/conv2d_2/Conv2D_1/ReadVariableOp?
+sig_ver_siamese_cnn/model/conv2d_2/Conv2D_1Conv2D<sig_ver_siamese_cnn/model/max_pooling2d_1/MaxPool_1:output:0Bsig_ver_siamese_cnn/model/conv2d_2/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????n7?*
paddingSAME*
strides
2-
+sig_ver_siamese_cnn/model/conv2d_2/Conv2D_1?
;sig_ver_siamese_cnn/model/conv2d_2/BiasAdd_1/ReadVariableOpReadVariableOpBsig_ver_siamese_cnn_model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02=
;sig_ver_siamese_cnn/model/conv2d_2/BiasAdd_1/ReadVariableOp?
,sig_ver_siamese_cnn/model/conv2d_2/BiasAdd_1BiasAdd4sig_ver_siamese_cnn/model/conv2d_2/Conv2D_1:output:0Csig_ver_siamese_cnn/model/conv2d_2/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????n7?2.
,sig_ver_siamese_cnn/model/conv2d_2/BiasAdd_1?
)sig_ver_siamese_cnn/model/conv2d_2/Relu_1Relu5sig_ver_siamese_cnn/model/conv2d_2/BiasAdd_1:output:0*
T0*0
_output_shapes
:?????????n7?2+
)sig_ver_siamese_cnn/model/conv2d_2/Relu_1?
3sig_ver_siamese_cnn/model/max_pooling2d_2/MaxPool_1MaxPool7sig_ver_siamese_cnn/model/conv2d_2/Relu_1:activations:0*0
_output_shapes
:?????????7?*
ksize
*
paddingVALID*
strides
25
3sig_ver_siamese_cnn/model/max_pooling2d_2/MaxPool_1?
:sig_ver_siamese_cnn/model/conv2d_3/Conv2D_1/ReadVariableOpReadVariableOpAsig_ver_siamese_cnn_model_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02<
:sig_ver_siamese_cnn/model/conv2d_3/Conv2D_1/ReadVariableOp?
+sig_ver_siamese_cnn/model/conv2d_3/Conv2D_1Conv2D<sig_ver_siamese_cnn/model/max_pooling2d_2/MaxPool_1:output:0Bsig_ver_siamese_cnn/model/conv2d_3/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????7?*
paddingSAME*
strides
2-
+sig_ver_siamese_cnn/model/conv2d_3/Conv2D_1?
;sig_ver_siamese_cnn/model/conv2d_3/BiasAdd_1/ReadVariableOpReadVariableOpBsig_ver_siamese_cnn_model_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02=
;sig_ver_siamese_cnn/model/conv2d_3/BiasAdd_1/ReadVariableOp?
,sig_ver_siamese_cnn/model/conv2d_3/BiasAdd_1BiasAdd4sig_ver_siamese_cnn/model/conv2d_3/Conv2D_1:output:0Csig_ver_siamese_cnn/model/conv2d_3/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????7?2.
,sig_ver_siamese_cnn/model/conv2d_3/BiasAdd_1?
)sig_ver_siamese_cnn/model/conv2d_3/Relu_1Relu5sig_ver_siamese_cnn/model/conv2d_3/BiasAdd_1:output:0*
T0*0
_output_shapes
:?????????7?2+
)sig_ver_siamese_cnn/model/conv2d_3/Relu_1?
3sig_ver_siamese_cnn/model/max_pooling2d_3/MaxPool_1MaxPool7sig_ver_siamese_cnn/model/conv2d_3/Relu_1:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
25
3sig_ver_siamese_cnn/model/max_pooling2d_3/MaxPool_1?
:sig_ver_siamese_cnn/model/conv2d_4/Conv2D_1/ReadVariableOpReadVariableOpAsig_ver_siamese_cnn_model_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02<
:sig_ver_siamese_cnn/model/conv2d_4/Conv2D_1/ReadVariableOp?
+sig_ver_siamese_cnn/model/conv2d_4/Conv2D_1Conv2D<sig_ver_siamese_cnn/model/max_pooling2d_3/MaxPool_1:output:0Bsig_ver_siamese_cnn/model/conv2d_4/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2-
+sig_ver_siamese_cnn/model/conv2d_4/Conv2D_1?
;sig_ver_siamese_cnn/model/conv2d_4/BiasAdd_1/ReadVariableOpReadVariableOpBsig_ver_siamese_cnn_model_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02=
;sig_ver_siamese_cnn/model/conv2d_4/BiasAdd_1/ReadVariableOp?
,sig_ver_siamese_cnn/model/conv2d_4/BiasAdd_1BiasAdd4sig_ver_siamese_cnn/model/conv2d_4/Conv2D_1:output:0Csig_ver_siamese_cnn/model/conv2d_4/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2.
,sig_ver_siamese_cnn/model/conv2d_4/BiasAdd_1?
)sig_ver_siamese_cnn/model/conv2d_4/Relu_1Relu5sig_ver_siamese_cnn/model/conv2d_4/BiasAdd_1:output:0*
T0*0
_output_shapes
:??????????2+
)sig_ver_siamese_cnn/model/conv2d_4/Relu_1?
.sig_ver_siamese_cnn/model/dropout_1/Identity_1Identity7sig_ver_siamese_cnn/model/conv2d_4/Relu_1:activations:0*
T0*0
_output_shapes
:??????????20
.sig_ver_siamese_cnn/model/dropout_1/Identity_1?
Ksig_ver_siamese_cnn/model/global_average_pooling2d/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2M
Ksig_ver_siamese_cnn/model/global_average_pooling2d/Mean_1/reduction_indices?
9sig_ver_siamese_cnn/model/global_average_pooling2d/Mean_1Mean7sig_ver_siamese_cnn/model/dropout_1/Identity_1:output:0Tsig_ver_siamese_cnn/model/global_average_pooling2d/Mean_1/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2;
9sig_ver_siamese_cnn/model/global_average_pooling2d/Mean_1?
7sig_ver_siamese_cnn/model/dense/MatMul_1/ReadVariableOpReadVariableOp>sig_ver_siamese_cnn_model_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype029
7sig_ver_siamese_cnn/model/dense/MatMul_1/ReadVariableOp?
(sig_ver_siamese_cnn/model/dense/MatMul_1MatMulBsig_ver_siamese_cnn/model/global_average_pooling2d/Mean_1:output:0?sig_ver_siamese_cnn/model/dense/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2*
(sig_ver_siamese_cnn/model/dense/MatMul_1?
8sig_ver_siamese_cnn/model/dense/BiasAdd_1/ReadVariableOpReadVariableOp?sig_ver_siamese_cnn_model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8sig_ver_siamese_cnn/model/dense/BiasAdd_1/ReadVariableOp?
)sig_ver_siamese_cnn/model/dense/BiasAdd_1BiasAdd2sig_ver_siamese_cnn/model/dense/MatMul_1:product:0@sig_ver_siamese_cnn/model/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2+
)sig_ver_siamese_cnn/model/dense/BiasAdd_1?
(sig_ver_siamese_cnn/tf.math.subtract/SubSub0sig_ver_siamese_cnn/model/dense/BiasAdd:output:02sig_ver_siamese_cnn/model/dense/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????2*
(sig_ver_siamese_cnn/tf.math.subtract/Sub?
)sig_ver_siamese_cnn/tf.math.square/SquareSquare,sig_ver_siamese_cnn/tf.math.subtract/Sub:z:0*
T0*'
_output_shapes
:?????????2+
)sig_ver_siamese_cnn/tf.math.square/Square?
<sig_ver_siamese_cnn/tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2>
<sig_ver_siamese_cnn/tf.math.reduce_sum/Sum/reduction_indices?
*sig_ver_siamese_cnn/tf.math.reduce_sum/SumSum-sig_ver_siamese_cnn/tf.math.square/Square:y:0Esig_ver_siamese_cnn/tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2,
*sig_ver_siamese_cnn/tf.math.reduce_sum/Sum?
:sig_ver_siamese_cnn/tf.clip_by_value/clip_by_value/MinimumMinimum3sig_ver_siamese_cnn/tf.math.reduce_sum/Sum:output:0<sig_ver_siamese_cnn_tf_clip_by_value_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????2<
:sig_ver_siamese_cnn/tf.clip_by_value/clip_by_value/Minimum?
2sig_ver_siamese_cnn/tf.clip_by_value/clip_by_valueMaximum>sig_ver_siamese_cnn/tf.clip_by_value/clip_by_value/Minimum:z:04sig_ver_siamese_cnn_tf_clip_by_value_clip_by_value_y*
T0*'
_output_shapes
:?????????24
2sig_ver_siamese_cnn/tf.clip_by_value/clip_by_value?
%sig_ver_siamese_cnn/tf.math.sqrt/SqrtSqrt6sig_ver_siamese_cnn/tf.clip_by_value/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2'
%sig_ver_siamese_cnn/tf.math.sqrt/Sqrt?
IdentityIdentity)sig_ver_siamese_cnn/tf.math.sqrt/Sqrt:y:08^sig_ver_siamese_cnn/model/conv2d/BiasAdd/ReadVariableOp:^sig_ver_siamese_cnn/model/conv2d/BiasAdd_1/ReadVariableOp7^sig_ver_siamese_cnn/model/conv2d/Conv2D/ReadVariableOp9^sig_ver_siamese_cnn/model/conv2d/Conv2D_1/ReadVariableOp:^sig_ver_siamese_cnn/model/conv2d_1/BiasAdd/ReadVariableOp<^sig_ver_siamese_cnn/model/conv2d_1/BiasAdd_1/ReadVariableOp9^sig_ver_siamese_cnn/model/conv2d_1/Conv2D/ReadVariableOp;^sig_ver_siamese_cnn/model/conv2d_1/Conv2D_1/ReadVariableOp:^sig_ver_siamese_cnn/model/conv2d_2/BiasAdd/ReadVariableOp<^sig_ver_siamese_cnn/model/conv2d_2/BiasAdd_1/ReadVariableOp9^sig_ver_siamese_cnn/model/conv2d_2/Conv2D/ReadVariableOp;^sig_ver_siamese_cnn/model/conv2d_2/Conv2D_1/ReadVariableOp:^sig_ver_siamese_cnn/model/conv2d_3/BiasAdd/ReadVariableOp<^sig_ver_siamese_cnn/model/conv2d_3/BiasAdd_1/ReadVariableOp9^sig_ver_siamese_cnn/model/conv2d_3/Conv2D/ReadVariableOp;^sig_ver_siamese_cnn/model/conv2d_3/Conv2D_1/ReadVariableOp:^sig_ver_siamese_cnn/model/conv2d_4/BiasAdd/ReadVariableOp<^sig_ver_siamese_cnn/model/conv2d_4/BiasAdd_1/ReadVariableOp9^sig_ver_siamese_cnn/model/conv2d_4/Conv2D/ReadVariableOp;^sig_ver_siamese_cnn/model/conv2d_4/Conv2D_1/ReadVariableOp7^sig_ver_siamese_cnn/model/dense/BiasAdd/ReadVariableOp9^sig_ver_siamese_cnn/model/dense/BiasAdd_1/ReadVariableOp6^sig_ver_siamese_cnn/model/dense/MatMul/ReadVariableOp8^sig_ver_siamese_cnn/model/dense/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesp
n:???????????:???????????::::::::::::: : 2r
7sig_ver_siamese_cnn/model/conv2d/BiasAdd/ReadVariableOp7sig_ver_siamese_cnn/model/conv2d/BiasAdd/ReadVariableOp2v
9sig_ver_siamese_cnn/model/conv2d/BiasAdd_1/ReadVariableOp9sig_ver_siamese_cnn/model/conv2d/BiasAdd_1/ReadVariableOp2p
6sig_ver_siamese_cnn/model/conv2d/Conv2D/ReadVariableOp6sig_ver_siamese_cnn/model/conv2d/Conv2D/ReadVariableOp2t
8sig_ver_siamese_cnn/model/conv2d/Conv2D_1/ReadVariableOp8sig_ver_siamese_cnn/model/conv2d/Conv2D_1/ReadVariableOp2v
9sig_ver_siamese_cnn/model/conv2d_1/BiasAdd/ReadVariableOp9sig_ver_siamese_cnn/model/conv2d_1/BiasAdd/ReadVariableOp2z
;sig_ver_siamese_cnn/model/conv2d_1/BiasAdd_1/ReadVariableOp;sig_ver_siamese_cnn/model/conv2d_1/BiasAdd_1/ReadVariableOp2t
8sig_ver_siamese_cnn/model/conv2d_1/Conv2D/ReadVariableOp8sig_ver_siamese_cnn/model/conv2d_1/Conv2D/ReadVariableOp2x
:sig_ver_siamese_cnn/model/conv2d_1/Conv2D_1/ReadVariableOp:sig_ver_siamese_cnn/model/conv2d_1/Conv2D_1/ReadVariableOp2v
9sig_ver_siamese_cnn/model/conv2d_2/BiasAdd/ReadVariableOp9sig_ver_siamese_cnn/model/conv2d_2/BiasAdd/ReadVariableOp2z
;sig_ver_siamese_cnn/model/conv2d_2/BiasAdd_1/ReadVariableOp;sig_ver_siamese_cnn/model/conv2d_2/BiasAdd_1/ReadVariableOp2t
8sig_ver_siamese_cnn/model/conv2d_2/Conv2D/ReadVariableOp8sig_ver_siamese_cnn/model/conv2d_2/Conv2D/ReadVariableOp2x
:sig_ver_siamese_cnn/model/conv2d_2/Conv2D_1/ReadVariableOp:sig_ver_siamese_cnn/model/conv2d_2/Conv2D_1/ReadVariableOp2v
9sig_ver_siamese_cnn/model/conv2d_3/BiasAdd/ReadVariableOp9sig_ver_siamese_cnn/model/conv2d_3/BiasAdd/ReadVariableOp2z
;sig_ver_siamese_cnn/model/conv2d_3/BiasAdd_1/ReadVariableOp;sig_ver_siamese_cnn/model/conv2d_3/BiasAdd_1/ReadVariableOp2t
8sig_ver_siamese_cnn/model/conv2d_3/Conv2D/ReadVariableOp8sig_ver_siamese_cnn/model/conv2d_3/Conv2D/ReadVariableOp2x
:sig_ver_siamese_cnn/model/conv2d_3/Conv2D_1/ReadVariableOp:sig_ver_siamese_cnn/model/conv2d_3/Conv2D_1/ReadVariableOp2v
9sig_ver_siamese_cnn/model/conv2d_4/BiasAdd/ReadVariableOp9sig_ver_siamese_cnn/model/conv2d_4/BiasAdd/ReadVariableOp2z
;sig_ver_siamese_cnn/model/conv2d_4/BiasAdd_1/ReadVariableOp;sig_ver_siamese_cnn/model/conv2d_4/BiasAdd_1/ReadVariableOp2t
8sig_ver_siamese_cnn/model/conv2d_4/Conv2D/ReadVariableOp8sig_ver_siamese_cnn/model/conv2d_4/Conv2D/ReadVariableOp2x
:sig_ver_siamese_cnn/model/conv2d_4/Conv2D_1/ReadVariableOp:sig_ver_siamese_cnn/model/conv2d_4/Conv2D_1/ReadVariableOp2p
6sig_ver_siamese_cnn/model/dense/BiasAdd/ReadVariableOp6sig_ver_siamese_cnn/model/dense/BiasAdd/ReadVariableOp2t
8sig_ver_siamese_cnn/model/dense/BiasAdd_1/ReadVariableOp8sig_ver_siamese_cnn/model/dense/BiasAdd_1/ReadVariableOp2n
5sig_ver_siamese_cnn/model/dense/MatMul/ReadVariableOp5sig_ver_siamese_cnn/model/dense/MatMul/ReadVariableOp2r
7sig_ver_siamese_cnn/model/dense/MatMul_1/ReadVariableOp7sig_ver_siamese_cnn/model/dense/MatMul_1/ReadVariableOp:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1:ZV
1
_output_shapes
:???????????
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: 
?
|
'__inference_conv2d_1_layer_call_fn_8706

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????n@*$
_read_only_resource_inputs
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_74452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????n@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????n ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????n 
 
_user_specified_nameinputs
?
b
C__inference_dropout_1_layer_call_and_return_conditional_losses_8778

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
"__inference_signature_wrapper_8165
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *(
f#R!
__inference__wrapped_model_73112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesp
n:???????????:???????????::::::::::::: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1:ZV
1
_output_shapes
:???????????
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: 
?

?
B__inference_conv2d_2_layer_call_and_return_conditional_losses_7473

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????n7?*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????n7?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????n7?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????n7?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????n7@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????n7@
 
_user_specified_nameinputs
?
J
.__inference_max_pooling2d_2_layer_call_fn_7347

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *8
config_proto(&

CPU

GPU2*0J

  ?E8? *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_73412
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
?
M__inference_sig_ver_siamese_cnn_layer_call_and_return_conditional_losses_8293
inputs_0
inputs_1/
+model_conv2d_conv2d_readvariableop_resource0
,model_conv2d_biasadd_readvariableop_resource1
-model_conv2d_1_conv2d_readvariableop_resource2
.model_conv2d_1_biasadd_readvariableop_resource1
-model_conv2d_2_conv2d_readvariableop_resource2
.model_conv2d_2_biasadd_readvariableop_resource1
-model_conv2d_3_conv2d_readvariableop_resource2
.model_conv2d_3_biasadd_readvariableop_resource1
-model_conv2d_4_conv2d_readvariableop_resource2
.model_conv2d_4_biasadd_readvariableop_resource.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource,
(tf_clip_by_value_clip_by_value_minimum_y$
 tf_clip_by_value_clip_by_value_y
identity??#model/conv2d/BiasAdd/ReadVariableOp?%model/conv2d/BiasAdd_1/ReadVariableOp?"model/conv2d/Conv2D/ReadVariableOp?$model/conv2d/Conv2D_1/ReadVariableOp?%model/conv2d_1/BiasAdd/ReadVariableOp?'model/conv2d_1/BiasAdd_1/ReadVariableOp?$model/conv2d_1/Conv2D/ReadVariableOp?&model/conv2d_1/Conv2D_1/ReadVariableOp?%model/conv2d_2/BiasAdd/ReadVariableOp?'model/conv2d_2/BiasAdd_1/ReadVariableOp?$model/conv2d_2/Conv2D/ReadVariableOp?&model/conv2d_2/Conv2D_1/ReadVariableOp?%model/conv2d_3/BiasAdd/ReadVariableOp?'model/conv2d_3/BiasAdd_1/ReadVariableOp?$model/conv2d_3/Conv2D/ReadVariableOp?&model/conv2d_3/Conv2D_1/ReadVariableOp?%model/conv2d_4/BiasAdd/ReadVariableOp?'model/conv2d_4/BiasAdd_1/ReadVariableOp?$model/conv2d_4/Conv2D/ReadVariableOp?&model/conv2d_4/Conv2D_1/ReadVariableOp?"model/dense/BiasAdd/ReadVariableOp?$model/dense/BiasAdd_1/ReadVariableOp?!model/dense/MatMul/ReadVariableOp?#model/dense/MatMul_1/ReadVariableOp?
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02$
"model/conv2d/Conv2D/ReadVariableOp?
model/conv2d/Conv2DConv2Dinputs_0*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
model/conv2d/Conv2D?
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#model/conv2d/BiasAdd/ReadVariableOp?
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
model/conv2d/BiasAdd?
model/conv2d/ReluRelumodel/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
model/conv2d/Relu?
model/max_pooling2d/MaxPoolMaxPoolmodel/conv2d/Relu:activations:0*0
_output_shapes
:??????????n *
ksize
*
paddingVALID*
strides
2
model/max_pooling2d/MaxPool
model/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
model/dropout/dropout/Const?
model/dropout/dropout/MulMul$model/max_pooling2d/MaxPool:output:0$model/dropout/dropout/Const:output:0*
T0*0
_output_shapes
:??????????n 2
model/dropout/dropout/Mul?
model/dropout/dropout/ShapeShape$model/max_pooling2d/MaxPool:output:0*
T0*
_output_shapes
:2
model/dropout/dropout/Shape?
2model/dropout/dropout/random_uniform/RandomUniformRandomUniform$model/dropout/dropout/Shape:output:0*
T0*0
_output_shapes
:??????????n *
dtype024
2model/dropout/dropout/random_uniform/RandomUniform?
$model/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$model/dropout/dropout/GreaterEqual/y?
"model/dropout/dropout/GreaterEqualGreaterEqual;model/dropout/dropout/random_uniform/RandomUniform:output:0-model/dropout/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????n 2$
"model/dropout/dropout/GreaterEqual?
model/dropout/dropout/CastCast&model/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????n 2
model/dropout/dropout/Cast?
model/dropout/dropout/Mul_1Mulmodel/dropout/dropout/Mul:z:0model/dropout/dropout/Cast:y:0*
T0*0
_output_shapes
:??????????n 2
model/dropout/dropout/Mul_1?
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02&
$model/conv2d_1/Conv2D/ReadVariableOp?
model/conv2d_1/Conv2DConv2Dmodel/dropout/dropout/Mul_1:z:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????n@*
paddingSAME*
strides
2
model/conv2d_1/Conv2D?
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model/conv2d_1/BiasAdd/ReadVariableOp?
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????n@2
model/conv2d_1/BiasAdd?
model/conv2d_1/ReluRelumodel/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????n@2
model/conv2d_1/Relu?
model/max_pooling2d_1/MaxPoolMaxPool!model/conv2d_1/Relu:activations:0*/
_output_shapes
:?????????n7@*
ksize
*
paddingVALID*
strides
2
model/max_pooling2d_1/MaxPool?
$model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02&
$model/conv2d_2/Conv2D/ReadVariableOp?
model/conv2d_2/Conv2DConv2D&model/max_pooling2d_1/MaxPool:output:0,model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????n7?*
paddingSAME*
strides
2
model/conv2d_2/Conv2D?
%model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%model/conv2d_2/BiasAdd/ReadVariableOp?
model/conv2d_2/BiasAddBiasAddmodel/conv2d_2/Conv2D:output:0-model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????n7?2
model/conv2d_2/BiasAdd?
model/conv2d_2/ReluRelumodel/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????n7?2
model/conv2d_2/Relu?
model/max_pooling2d_2/MaxPoolMaxPool!model/conv2d_2/Relu:activations:0*0
_output_shapes
:?????????7?*
ksize
*
paddingVALID*
strides
2
model/max_pooling2d_2/MaxPool?
$model/conv2d_3/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02&
$model/conv2d_3/Conv2D/ReadVariableOp?
model/conv2d_3/Conv2DConv2D&model/max_pooling2d_2/MaxPool:output:0,model/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????7?*
paddingSAME*
strides
2
model/conv2d_3/Conv2D?
%model/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%model/conv2d_3/BiasAdd/ReadVariableOp?
model/conv2d_3/BiasAddBiasAddmodel/conv2d_3/Conv2D:output:0-model/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????7?2
model/conv2d_3/BiasAdd?
model/conv2d_3/ReluRelumodel/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????7?2
model/conv2d_3/Relu?
model/max_pooling2d_3/MaxPoolMaxPool!model/conv2d_3/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2
model/max_pooling2d_3/MaxPool?
$model/conv2d_4/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02&
$model/conv2d_4/Conv2D/ReadVariableOp?
model/conv2d_4/Conv2DConv2D&model/max_pooling2d_3/MaxPool:output:0,model/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model/conv2d_4/Conv2D?
%model/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%model/conv2d_4/BiasAdd/ReadVariableOp?
model/conv2d_4/BiasAddBiasAddmodel/conv2d_4/Conv2D:output:0-model/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model/conv2d_4/BiasAdd?
model/conv2d_4/ReluRelumodel/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model/conv2d_4/Relu?
model/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
model/dropout_1/dropout/Const?
model/dropout_1/dropout/MulMul!model/conv2d_4/Relu:activations:0&model/dropout_1/dropout/Const:output:0*
T0*0
_output_shapes
:??????????2
model/dropout_1/dropout/Mul?
model/dropout_1/dropout/ShapeShape!model/conv2d_4/Relu:activations:0*
T0*
_output_shapes
:2
model/dropout_1/dropout/Shape?
4model/dropout_1/dropout/random_uniform/RandomUniformRandomUniform&model/dropout_1/dropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype026
4model/dropout_1/dropout/random_uniform/RandomUniform?
&model/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2(
&model/dropout_1/dropout/GreaterEqual/y?
$model/dropout_1/dropout/GreaterEqualGreaterEqual=model/dropout_1/dropout/random_uniform/RandomUniform:output:0/model/dropout_1/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2&
$model/dropout_1/dropout/GreaterEqual?
model/dropout_1/dropout/CastCast(model/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2
model/dropout_1/dropout/Cast?
model/dropout_1/dropout/Mul_1Mulmodel/dropout_1/dropout/Mul:z:0 model/dropout_1/dropout/Cast:y:0*
T0*0
_output_shapes
:??????????2
model/dropout_1/dropout/Mul_1?
5model/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      27
5model/global_average_pooling2d/Mean/reduction_indices?
#model/global_average_pooling2d/MeanMean!model/dropout_1/dropout/Mul_1:z:0>model/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2%
#model/global_average_pooling2d/Mean?
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!model/dense/MatMul/ReadVariableOp?
model/dense/MatMulMatMul,model/global_average_pooling2d/Mean:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense/MatMul?
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/dense/BiasAdd/ReadVariableOp?
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense/BiasAdd?
$model/conv2d/Conv2D_1/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02&
$model/conv2d/Conv2D_1/ReadVariableOp?
model/conv2d/Conv2D_1Conv2Dinputs_1,model/conv2d/Conv2D_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
2
model/conv2d/Conv2D_1?
%model/conv2d/BiasAdd_1/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%model/conv2d/BiasAdd_1/ReadVariableOp?
model/conv2d/BiasAdd_1BiasAddmodel/conv2d/Conv2D_1:output:0-model/conv2d/BiasAdd_1/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
model/conv2d/BiasAdd_1?
model/conv2d/Relu_1Relumodel/conv2d/BiasAdd_1:output:0*
T0*1
_output_shapes
:??????????? 2
model/conv2d/Relu_1?
model/max_pooling2d/MaxPool_1MaxPool!model/conv2d/Relu_1:activations:0*0
_output_shapes
:??????????n *
ksize
*
paddingVALID*
strides
2
model/max_pooling2d/MaxPool_1?
model/dropout/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
model/dropout/dropout_1/Const?
model/dropout/dropout_1/MulMul&model/max_pooling2d/MaxPool_1:output:0&model/dropout/dropout_1/Const:output:0*
T0*0
_output_shapes
:??????????n 2
model/dropout/dropout_1/Mul?
model/dropout/dropout_1/ShapeShape&model/max_pooling2d/MaxPool_1:output:0*
T0*
_output_shapes
:2
model/dropout/dropout_1/Shape?
4model/dropout/dropout_1/random_uniform/RandomUniformRandomUniform&model/dropout/dropout_1/Shape:output:0*
T0*0
_output_shapes
:??????????n *
dtype026
4model/dropout/dropout_1/random_uniform/RandomUniform?
&model/dropout/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2(
&model/dropout/dropout_1/GreaterEqual/y?
$model/dropout/dropout_1/GreaterEqualGreaterEqual=model/dropout/dropout_1/random_uniform/RandomUniform:output:0/model/dropout/dropout_1/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????n 2&
$model/dropout/dropout_1/GreaterEqual?
model/dropout/dropout_1/CastCast(model/dropout/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????n 2
model/dropout/dropout_1/Cast?
model/dropout/dropout_1/Mul_1Mulmodel/dropout/dropout_1/Mul:z:0 model/dropout/dropout_1/Cast:y:0*
T0*0
_output_shapes
:??????????n 2
model/dropout/dropout_1/Mul_1?
&model/conv2d_1/Conv2D_1/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02(
&model/conv2d_1/Conv2D_1/ReadVariableOp?
model/conv2d_1/Conv2D_1Conv2D!model/dropout/dropout_1/Mul_1:z:0.model/conv2d_1/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????n@*
paddingSAME*
strides
2
model/conv2d_1/Conv2D_1?
'model/conv2d_1/BiasAdd_1/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model/conv2d_1/BiasAdd_1/ReadVariableOp?
model/conv2d_1/BiasAdd_1BiasAdd model/conv2d_1/Conv2D_1:output:0/model/conv2d_1/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????n@2
model/conv2d_1/BiasAdd_1?
model/conv2d_1/Relu_1Relu!model/conv2d_1/BiasAdd_1:output:0*
T0*0
_output_shapes
:??????????n@2
model/conv2d_1/Relu_1?
model/max_pooling2d_1/MaxPool_1MaxPool#model/conv2d_1/Relu_1:activations:0*/
_output_shapes
:?????????n7@*
ksize
*
paddingVALID*
strides
2!
model/max_pooling2d_1/MaxPool_1?
&model/conv2d_2/Conv2D_1/ReadVariableOpReadVariableOp-model_conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02(
&model/conv2d_2/Conv2D_1/ReadVariableOp?
model/conv2d_2/Conv2D_1Conv2D(model/max_pooling2d_1/MaxPool_1:output:0.model/conv2d_2/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????n7?*
paddingSAME*
strides
2
model/conv2d_2/Conv2D_1?
'model/conv2d_2/BiasAdd_1/ReadVariableOpReadVariableOp.model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'model/conv2d_2/BiasAdd_1/ReadVariableOp?
model/conv2d_2/BiasAdd_1BiasAdd model/conv2d_2/Conv2D_1:output:0/model/conv2d_2/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????n7?2
model/conv2d_2/BiasAdd_1?
model/conv2d_2/Relu_1Relu!model/conv2d_2/BiasAdd_1:output:0*
T0*0
_output_shapes
:?????????n7?2
model/conv2d_2/Relu_1?
model/max_pooling2d_2/MaxPool_1MaxPool#model/conv2d_2/Relu_1:activations:0*0
_output_shapes
:?????????7?*
ksize
*
paddingVALID*
strides
2!
model/max_pooling2d_2/MaxPool_1?
&model/conv2d_3/Conv2D_1/ReadVariableOpReadVariableOp-model_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&model/conv2d_3/Conv2D_1/ReadVariableOp?
model/conv2d_3/Conv2D_1Conv2D(model/max_pooling2d_2/MaxPool_1:output:0.model/conv2d_3/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????7?*
paddingSAME*
strides
2
model/conv2d_3/Conv2D_1?
'model/conv2d_3/BiasAdd_1/ReadVariableOpReadVariableOp.model_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'model/conv2d_3/BiasAdd_1/ReadVariableOp?
model/conv2d_3/BiasAdd_1BiasAdd model/conv2d_3/Conv2D_1:output:0/model/conv2d_3/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????7?2
model/conv2d_3/BiasAdd_1?
model/conv2d_3/Relu_1Relu!model/conv2d_3/BiasAdd_1:output:0*
T0*0
_output_shapes
:?????????7?2
model/conv2d_3/Relu_1?
model/max_pooling2d_3/MaxPool_1MaxPool#model/conv2d_3/Relu_1:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
2!
model/max_pooling2d_3/MaxPool_1?
&model/conv2d_4/Conv2D_1/ReadVariableOpReadVariableOp-model_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&model/conv2d_4/Conv2D_1/ReadVariableOp?
model/conv2d_4/Conv2D_1Conv2D(model/max_pooling2d_3/MaxPool_1:output:0.model/conv2d_4/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model/conv2d_4/Conv2D_1?
'model/conv2d_4/BiasAdd_1/ReadVariableOpReadVariableOp.model_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'model/conv2d_4/BiasAdd_1/ReadVariableOp?
model/conv2d_4/BiasAdd_1BiasAdd model/conv2d_4/Conv2D_1:output:0/model/conv2d_4/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model/conv2d_4/BiasAdd_1?
model/conv2d_4/Relu_1Relu!model/conv2d_4/BiasAdd_1:output:0*
T0*0
_output_shapes
:??????????2
model/conv2d_4/Relu_1?
model/dropout_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2!
model/dropout_1/dropout_1/Const?
model/dropout_1/dropout_1/MulMul#model/conv2d_4/Relu_1:activations:0(model/dropout_1/dropout_1/Const:output:0*
T0*0
_output_shapes
:??????????2
model/dropout_1/dropout_1/Mul?
model/dropout_1/dropout_1/ShapeShape#model/conv2d_4/Relu_1:activations:0*
T0*
_output_shapes
:2!
model/dropout_1/dropout_1/Shape?
6model/dropout_1/dropout_1/random_uniform/RandomUniformRandomUniform(model/dropout_1/dropout_1/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype028
6model/dropout_1/dropout_1/random_uniform/RandomUniform?
(model/dropout_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2*
(model/dropout_1/dropout_1/GreaterEqual/y?
&model/dropout_1/dropout_1/GreaterEqualGreaterEqual?model/dropout_1/dropout_1/random_uniform/RandomUniform:output:01model/dropout_1/dropout_1/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????2(
&model/dropout_1/dropout_1/GreaterEqual?
model/dropout_1/dropout_1/CastCast*model/dropout_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????2 
model/dropout_1/dropout_1/Cast?
model/dropout_1/dropout_1/Mul_1Mul!model/dropout_1/dropout_1/Mul:z:0"model/dropout_1/dropout_1/Cast:y:0*
T0*0
_output_shapes
:??????????2!
model/dropout_1/dropout_1/Mul_1?
7model/global_average_pooling2d/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      29
7model/global_average_pooling2d/Mean_1/reduction_indices?
%model/global_average_pooling2d/Mean_1Mean#model/dropout_1/dropout_1/Mul_1:z:0@model/global_average_pooling2d/Mean_1/reduction_indices:output:0*
T0*(
_output_shapes
:??????????2'
%model/global_average_pooling2d/Mean_1?
#model/dense/MatMul_1/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02%
#model/dense/MatMul_1/ReadVariableOp?
model/dense/MatMul_1MatMul.model/global_average_pooling2d/Mean_1:output:0+model/dense/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense/MatMul_1?
$model/dense/BiasAdd_1/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense/BiasAdd_1/ReadVariableOp?
model/dense/BiasAdd_1BiasAddmodel/dense/MatMul_1:product:0,model/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense/BiasAdd_1?
tf.math.subtract/SubSubmodel/dense/BiasAdd:output:0model/dense/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract/Sub?
tf.math.square/SquareSquaretf.math.subtract/Sub:z:0*
T0*'
_output_shapes
:?????????2
tf.math.square/Square?
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2*
(tf.math.reduce_sum/Sum/reduction_indices?
tf.math.reduce_sum/SumSumtf.math.square/Square:y:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
tf.math.reduce_sum/Sum?
&tf.clip_by_value/clip_by_value/MinimumMinimumtf.math.reduce_sum/Sum:output:0(tf_clip_by_value_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????2(
&tf.clip_by_value/clip_by_value/Minimum?
tf.clip_by_value/clip_by_valueMaximum*tf.clip_by_value/clip_by_value/Minimum:z:0 tf_clip_by_value_clip_by_value_y*
T0*'
_output_shapes
:?????????2 
tf.clip_by_value/clip_by_value?
tf.math.sqrt/SqrtSqrt"tf.clip_by_value/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
tf.math.sqrt/Sqrt?
IdentityIdentitytf.math.sqrt/Sqrt:y:0$^model/conv2d/BiasAdd/ReadVariableOp&^model/conv2d/BiasAdd_1/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp%^model/conv2d/Conv2D_1/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp(^model/conv2d_1/BiasAdd_1/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp'^model/conv2d_1/Conv2D_1/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp(^model/conv2d_2/BiasAdd_1/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp'^model/conv2d_2/Conv2D_1/ReadVariableOp&^model/conv2d_3/BiasAdd/ReadVariableOp(^model/conv2d_3/BiasAdd_1/ReadVariableOp%^model/conv2d_3/Conv2D/ReadVariableOp'^model/conv2d_3/Conv2D_1/ReadVariableOp&^model/conv2d_4/BiasAdd/ReadVariableOp(^model/conv2d_4/BiasAdd_1/ReadVariableOp%^model/conv2d_4/Conv2D/ReadVariableOp'^model/conv2d_4/Conv2D_1/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp%^model/dense/BiasAdd_1/ReadVariableOp"^model/dense/MatMul/ReadVariableOp$^model/dense/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesp
n:???????????:???????????::::::::::::: : 2J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2N
%model/conv2d/BiasAdd_1/ReadVariableOp%model/conv2d/BiasAdd_1/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2L
$model/conv2d/Conv2D_1/ReadVariableOp$model/conv2d/Conv2D_1/ReadVariableOp2N
%model/conv2d_1/BiasAdd/ReadVariableOp%model/conv2d_1/BiasAdd/ReadVariableOp2R
'model/conv2d_1/BiasAdd_1/ReadVariableOp'model/conv2d_1/BiasAdd_1/ReadVariableOp2L
$model/conv2d_1/Conv2D/ReadVariableOp$model/conv2d_1/Conv2D/ReadVariableOp2P
&model/conv2d_1/Conv2D_1/ReadVariableOp&model/conv2d_1/Conv2D_1/ReadVariableOp2N
%model/conv2d_2/BiasAdd/ReadVariableOp%model/conv2d_2/BiasAdd/ReadVariableOp2R
'model/conv2d_2/BiasAdd_1/ReadVariableOp'model/conv2d_2/BiasAdd_1/ReadVariableOp2L
$model/conv2d_2/Conv2D/ReadVariableOp$model/conv2d_2/Conv2D/ReadVariableOp2P
&model/conv2d_2/Conv2D_1/ReadVariableOp&model/conv2d_2/Conv2D_1/ReadVariableOp2N
%model/conv2d_3/BiasAdd/ReadVariableOp%model/conv2d_3/BiasAdd/ReadVariableOp2R
'model/conv2d_3/BiasAdd_1/ReadVariableOp'model/conv2d_3/BiasAdd_1/ReadVariableOp2L
$model/conv2d_3/Conv2D/ReadVariableOp$model/conv2d_3/Conv2D/ReadVariableOp2P
&model/conv2d_3/Conv2D_1/ReadVariableOp&model/conv2d_3/Conv2D_1/ReadVariableOp2N
%model/conv2d_4/BiasAdd/ReadVariableOp%model/conv2d_4/BiasAdd/ReadVariableOp2R
'model/conv2d_4/BiasAdd_1/ReadVariableOp'model/conv2d_4/BiasAdd_1/ReadVariableOp2L
$model/conv2d_4/Conv2D/ReadVariableOp$model/conv2d_4/Conv2D/ReadVariableOp2P
&model/conv2d_4/Conv2D_1/ReadVariableOp&model/conv2d_4/Conv2D_1/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2L
$model/dense/BiasAdd_1/ReadVariableOp$model/dense/BiasAdd_1/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2J
#model/dense/MatMul_1/ReadVariableOp#model/dense/MatMul_1/ReadVariableOp:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: 
?
?
2__inference_sig_ver_siamese_cnn_layer_call_fn_8121
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *V
fQRO
M__inference_sig_ver_siamese_cnn_layer_call_and_return_conditional_losses_80902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesp
n:???????????:???????????::::::::::::: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1:ZV
1
_output_shapes
:???????????
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: 
?
e
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_7353

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
2__inference_sig_ver_siamese_cnn_layer_call_fn_8035
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *V
fQRO
M__inference_sig_ver_siamese_cnn_layer_call_and_return_conditional_losses_80042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesp
n:???????????:???????????::::::::::::: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1:ZV
1
_output_shapes
:???????????
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: 
?
?
M__inference_sig_ver_siamese_cnn_layer_call_and_return_conditional_losses_8090

inputs
inputs_1

model_8042

model_8044

model_8046

model_8048

model_8050

model_8052

model_8054

model_8056

model_8058

model_8060

model_8062

model_8064,
(tf_clip_by_value_clip_by_value_minimum_y$
 tf_clip_by_value_clip_by_value_y
identity??model/StatefulPartitionedCall?model/StatefulPartitionedCall_1?
model/StatefulPartitionedCallStatefulPartitionedCallinputs
model_8042
model_8044
model_8046
model_8048
model_8050
model_8052
model_8054
model_8056
model_8058
model_8060
model_8062
model_8064*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_77582
model/StatefulPartitionedCall?
model/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1
model_8042
model_8044
model_8046
model_8048
model_8050
model_8052
model_8054
model_8056
model_8058
model_8060
model_8062
model_8064*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*8
config_proto(&

CPU

GPU2*0J

  ?E8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_77582!
model/StatefulPartitionedCall_1?
tf.math.subtract/SubSub&model/StatefulPartitionedCall:output:0(model/StatefulPartitionedCall_1:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract/Sub?
tf.math.square/SquareSquaretf.math.subtract/Sub:z:0*
T0*'
_output_shapes
:?????????2
tf.math.square/Square?
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2*
(tf.math.reduce_sum/Sum/reduction_indices?
tf.math.reduce_sum/SumSumtf.math.square/Square:y:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
tf.math.reduce_sum/Sum?
&tf.clip_by_value/clip_by_value/MinimumMinimumtf.math.reduce_sum/Sum:output:0(tf_clip_by_value_clip_by_value_minimum_y*
T0*'
_output_shapes
:?????????2(
&tf.clip_by_value/clip_by_value/Minimum?
tf.clip_by_value/clip_by_valueMaximum*tf.clip_by_value/clip_by_value/Minimum:z:0 tf_clip_by_value_clip_by_value_y*
T0*'
_output_shapes
:?????????2 
tf.clip_by_value/clip_by_value?
tf.math.sqrt/SqrtSqrt"tf.clip_by_value/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
tf.math.sqrt/Sqrt?
IdentityIdentitytf.math.sqrt/Sqrt:y:0^model/StatefulPartitionedCall ^model/StatefulPartitionedCall_1*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesp
n:???????????:???????????::::::::::::: : 2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model/StatefulPartitionedCall_1model/StatefulPartitionedCall_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs:YU
1
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
`
A__inference_dropout_layer_call_and_return_conditional_losses_8671

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:??????????n 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????n *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????n 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????n 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????n 2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:??????????n 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????n :X T
0
_output_shapes
:??????????n 
 
_user_specified_nameinputs
?
_
A__inference_dropout_layer_call_and_return_conditional_losses_8676

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????n 2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????n 2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:??????????n :X T
0
_output_shapes
:??????????n 
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_1:
serving_default_input_1:0???????????
E
input_2:
serving_default_input_2:0???????????@
tf.math.sqrt0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
˒
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
		optimizer

trainable_variables
	variables
regularization_losses
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"??
_tf_keras_networkݏ{"class_name": "SigVerSiameseCNN", "name": "sig_ver_siamese_cnn", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sig_ver_siamese_cnn", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 440, 220, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 440, 220, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 440, 220, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [11, 11]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_2", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["max_pooling2d_2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_3", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["max_pooling2d_3", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["global_average_pooling2d", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["dense", 0, 0]]}, "name": "model", "inbound_nodes": [[["input_1", 0, 0, {}]], [["input_2", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract", "inbound_nodes": [["model", 1, 0, {"y": ["model", 2, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.square", "trainable": true, "dtype": "float32", "function": "math.square"}, "name": "tf.math.square", "inbound_nodes": [["tf.math.subtract", 0, 0, {"name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum", "inbound_nodes": [["tf.math.square", 0, 0, {"axis": 1, "keepdims": true}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value", "inbound_nodes": [["tf.math.reduce_sum", 0, 0, {"clip_value_min": 0.0, "clip_value_max": Infinity}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.sqrt", "trainable": true, "dtype": "float32", "function": "math.sqrt"}, "name": "tf.math.sqrt", "inbound_nodes": [["tf.clip_by_value", 0, 0, {}]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0]], "output_layers": [["tf.math.sqrt", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 440, 220, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 440, 220, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 440, 220, 1]}, {"class_name": "TensorShape", "items": [null, 440, 220, 1]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "SigVerSiameseCNN", "config": {"name": "sig_ver_siamese_cnn", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 440, 220, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 440, 220, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 440, 220, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [11, 11]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_2", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["max_pooling2d_2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_3", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["max_pooling2d_3", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["global_average_pooling2d", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["dense", 0, 0]]}, "name": "model", "inbound_nodes": [[["input_1", 0, 0, {}]], [["input_2", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract", "inbound_nodes": [["model", 1, 0, {"y": ["model", 2, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.square", "trainable": true, "dtype": "float32", "function": "math.square"}, "name": "tf.math.square", "inbound_nodes": [["tf.math.subtract", 0, 0, {"name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.reduce_sum", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}, "name": "tf.math.reduce_sum", "inbound_nodes": [["tf.math.square", 0, 0, {"axis": 1, "keepdims": true}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.clip_by_value", "trainable": true, "dtype": "float32", "function": "clip_by_value"}, "name": "tf.clip_by_value", "inbound_nodes": [["tf.math.reduce_sum", 0, 0, {"clip_value_min": 0.0, "clip_value_max": Infinity}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.sqrt", "trainable": true, "dtype": "float32", "function": "math.sqrt"}, "name": "tf.math.sqrt", "inbound_nodes": [["tf.clip_by_value", 0, 0, {}]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0]], "output_layers": [["tf.math.sqrt", 0, 0]]}}, "training_config": {"loss": {"class_name": "Addons>ContrastiveLoss", "config": {"reduction": "sum_over_batch_size", "name": "contrastive_loss", "margin": 1}}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 440, 220, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 440, 220, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 440, 220, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 440, 220, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
?p
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
layer_with_weights-3
layer-8
layer-9
layer_with_weights-4
layer-10
layer-11
layer-12
layer_with_weights-5
layer-13
trainable_variables
	variables
regularization_losses
 	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?l
_tf_keras_network?l{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 440, 220, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [11, 11]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_2", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["max_pooling2d_2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_3", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["max_pooling2d_3", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["global_average_pooling2d", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["dense", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 440, 220, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 440, 220, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 440, 220, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [11, 11]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_2", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["max_pooling2d_2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_3", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["max_pooling2d_3", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["global_average_pooling2d", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["dense", 0, 0]]}}}
?
!	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.subtract", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.subtract", "trainable": true, "dtype": "float32", "function": "math.subtract"}}
?
"	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.square", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.square", "trainable": true, "dtype": "float32", "function": "math.square"}}
?
#	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.reduce_sum", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.reduce_sum", "trainable": true, "dtype": "float32", "function": "math.reduce_sum"}}
?
$	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.clip_by_value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.clip_by_value", "trainable": true, "dtype": "float32", "function": "clip_by_value"}}
?
%	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.sqrt", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.sqrt", "trainable": true, "dtype": "float32", "function": "math.sqrt"}}
?
&iter

'beta_1

(beta_2
	)decay
*learning_rate+m?,m?-m?.m?/m?0m?1m?2m?3m?4m?5m?6m?+v?,v?-v?.v?/v?0v?1v?2v?3v?4v?5v?6v?"
	optimizer
v
+0
,1
-2
.3
/4
05
16
27
38
49
510
611"
trackable_list_wrapper
v
+0
,1
-2
.3
/4
05
16
27
38
49
510
611"
trackable_list_wrapper
 "
trackable_list_wrapper
?
7non_trainable_variables

trainable_variables

8layers
9metrics
:layer_metrics
	variables
regularization_losses
;layer_regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 440, 220, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 440, 220, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}
?	

+kernel
,bias
<trainable_variables
=	variables
>regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [11, 11]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 440, 220, 1]}}
?
@trainable_variables
A	variables
Bregularization_losses
C	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
Dtrainable_variables
E	variables
Fregularization_losses
G	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}
?	

-kernel
.bias
Htrainable_variables
I	variables
Jregularization_losses
K	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 220, 110, 32]}}
?
Ltrainable_variables
M	variables
Nregularization_losses
O	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

/kernel
0bias
Ptrainable_variables
Q	variables
Rregularization_losses
S	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 110, 55, 64]}}
?
Ttrainable_variables
U	variables
Vregularization_losses
W	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

1kernel
2bias
Xtrainable_variables
Y	variables
Zregularization_losses
[	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 55, 27, 128]}}
?
\trainable_variables
]	variables
^regularization_losses
_	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

3kernel
4bias
`trainable_variables
a	variables
bregularization_losses
c	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 27, 13, 256]}}
?
dtrainable_variables
e	variables
fregularization_losses
g	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}
?
htrainable_variables
i	variables
jregularization_losses
k	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "GlobalAveragePooling2D", "name": "global_average_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?

5kernel
6bias
ltrainable_variables
m	variables
nregularization_losses
o	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
v
+0
,1
-2
.3
/4
05
16
27
38
49
510
611"
trackable_list_wrapper
v
+0
,1
-2
.3
/4
05
16
27
38
49
510
611"
trackable_list_wrapper
 "
trackable_list_wrapper
?
pnon_trainable_variables
trainable_variables

qlayers
rmetrics
slayer_metrics
	variables
regularization_losses
tlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
':% 2conv2d/kernel
: 2conv2d/bias
):' @2conv2d_1/kernel
:@2conv2d_1/bias
*:(@?2conv2d_2/kernel
:?2conv2d_2/bias
+:)??2conv2d_3/kernel
:?2conv2d_3/bias
+:)??2conv2d_4/kernel
:?2conv2d_4/bias
:	?2dense/kernel
:2
dense/bias
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
'
u0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
vnon_trainable_variables
<trainable_variables

wlayers
xmetrics
ylayer_metrics
=	variables
>regularization_losses
zlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
{non_trainable_variables
@trainable_variables

|layers
}metrics
~layer_metrics
A	variables
Bregularization_losses
layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
Dtrainable_variables
?layers
?metrics
?layer_metrics
E	variables
Fregularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
Htrainable_variables
?layers
?metrics
?layer_metrics
I	variables
Jregularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
Ltrainable_variables
?layers
?metrics
?layer_metrics
M	variables
Nregularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
Ptrainable_variables
?layers
?metrics
?layer_metrics
Q	variables
Rregularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
Ttrainable_variables
?layers
?metrics
?layer_metrics
U	variables
Vregularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
Xtrainable_variables
?layers
?metrics
?layer_metrics
Y	variables
Zregularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
\trainable_variables
?layers
?metrics
?layer_metrics
]	variables
^regularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
`trainable_variables
?layers
?metrics
?layer_metrics
a	variables
bregularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
dtrainable_variables
?layers
?metrics
?layer_metrics
e	variables
fregularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
htrainable_variables
?layers
?metrics
?layer_metrics
i	variables
jregularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
ltrainable_variables
?layers
?metrics
?layer_metrics
m	variables
nregularization_losses
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
8
9
10
11
12
13"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
,:* 2Adam/conv2d/kernel/m
: 2Adam/conv2d/bias/m
.:, @2Adam/conv2d_1/kernel/m
 :@2Adam/conv2d_1/bias/m
/:-@?2Adam/conv2d_2/kernel/m
!:?2Adam/conv2d_2/bias/m
0:.??2Adam/conv2d_3/kernel/m
!:?2Adam/conv2d_3/bias/m
0:.??2Adam/conv2d_4/kernel/m
!:?2Adam/conv2d_4/bias/m
$:"	?2Adam/dense/kernel/m
:2Adam/dense/bias/m
,:* 2Adam/conv2d/kernel/v
: 2Adam/conv2d/bias/v
.:, @2Adam/conv2d_1/kernel/v
 :@2Adam/conv2d_1/bias/v
/:-@?2Adam/conv2d_2/kernel/v
!:?2Adam/conv2d_2/bias/v
0:.??2Adam/conv2d_3/kernel/v
!:?2Adam/conv2d_3/bias/v
0:.??2Adam/conv2d_4/kernel/v
!:?2Adam/conv2d_4/bias/v
$:"	?2Adam/dense/kernel/v
:2Adam/dense/bias/v
?2?
M__inference_sig_ver_siamese_cnn_layer_call_and_return_conditional_losses_8293
M__inference_sig_ver_siamese_cnn_layer_call_and_return_conditional_losses_8393
M__inference_sig_ver_siamese_cnn_layer_call_and_return_conditional_losses_7896
M__inference_sig_ver_siamese_cnn_layer_call_and_return_conditional_losses_7948?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference__wrapped_model_7311?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *b?_
]?Z
+?(
input_1???????????
+?(
input_2???????????
?2?
2__inference_sig_ver_siamese_cnn_layer_call_fn_8121
2__inference_sig_ver_siamese_cnn_layer_call_fn_8461
2__inference_sig_ver_siamese_cnn_layer_call_fn_8035
2__inference_sig_ver_siamese_cnn_layer_call_fn_8427?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
?__inference_model_layer_call_and_return_conditional_losses_8528
?__inference_model_layer_call_and_return_conditional_losses_8581
?__inference_model_layer_call_and_return_conditional_losses_7603
?__inference_model_layer_call_and_return_conditional_losses_7644?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
$__inference_model_layer_call_fn_8639
$__inference_model_layer_call_fn_7715
$__inference_model_layer_call_fn_8610
$__inference_model_layer_call_fn_7785?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
"__inference_signature_wrapper_8165input_1input_2"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_conv2d_layer_call_and_return_conditional_losses_8650?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_conv2d_layer_call_fn_8659?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_7317?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
,__inference_max_pooling2d_layer_call_fn_7323?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
A__inference_dropout_layer_call_and_return_conditional_losses_8676
A__inference_dropout_layer_call_and_return_conditional_losses_8671?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_dropout_layer_call_fn_8686
&__inference_dropout_layer_call_fn_8681?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_conv2d_1_layer_call_and_return_conditional_losses_8697?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv2d_1_layer_call_fn_8706?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_7329?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
.__inference_max_pooling2d_1_layer_call_fn_7335?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
B__inference_conv2d_2_layer_call_and_return_conditional_losses_8717?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv2d_2_layer_call_fn_8726?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_7341?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
.__inference_max_pooling2d_2_layer_call_fn_7347?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
B__inference_conv2d_3_layer_call_and_return_conditional_losses_8737?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv2d_3_layer_call_fn_8746?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_7353?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
.__inference_max_pooling2d_3_layer_call_fn_7359?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
B__inference_conv2d_4_layer_call_and_return_conditional_losses_8757?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv2d_4_layer_call_fn_8766?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dropout_1_layer_call_and_return_conditional_losses_8778
C__inference_dropout_1_layer_call_and_return_conditional_losses_8783?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dropout_1_layer_call_fn_8788
(__inference_dropout_1_layer_call_fn_8793?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_7366?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
7__inference_global_average_pooling2d_layer_call_fn_7372?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
?__inference_dense_layer_call_and_return_conditional_losses_8803?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_dense_layer_call_fn_8812?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
	J
Const
J	
Const_1?
__inference__wrapped_model_7311?+,-./0123456??l?i
b?_
]?Z
+?(
input_1???????????
+?(
input_2???????????
? ";?8
6
tf.math.sqrt&?#
tf.math.sqrt??????????
B__inference_conv2d_1_layer_call_and_return_conditional_losses_8697n-.8?5
.?+
)?&
inputs??????????n 
? ".?+
$?!
0??????????n@
? ?
'__inference_conv2d_1_layer_call_fn_8706a-.8?5
.?+
)?&
inputs??????????n 
? "!???????????n@?
B__inference_conv2d_2_layer_call_and_return_conditional_losses_8717m/07?4
-?*
(?%
inputs?????????n7@
? ".?+
$?!
0?????????n7?
? ?
'__inference_conv2d_2_layer_call_fn_8726`/07?4
-?*
(?%
inputs?????????n7@
? "!??????????n7??
B__inference_conv2d_3_layer_call_and_return_conditional_losses_8737n128?5
.?+
)?&
inputs?????????7?
? ".?+
$?!
0?????????7?
? ?
'__inference_conv2d_3_layer_call_fn_8746a128?5
.?+
)?&
inputs?????????7?
? "!??????????7??
B__inference_conv2d_4_layer_call_and_return_conditional_losses_8757n348?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
'__inference_conv2d_4_layer_call_fn_8766a348?5
.?+
)?&
inputs??????????
? "!????????????
@__inference_conv2d_layer_call_and_return_conditional_losses_8650p+,9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0??????????? 
? ?
%__inference_conv2d_layer_call_fn_8659c+,9?6
/?,
*?'
inputs???????????
? ""???????????? ?
?__inference_dense_layer_call_and_return_conditional_losses_8803]560?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? x
$__inference_dense_layer_call_fn_8812P560?-
&?#
!?
inputs??????????
? "???????????
C__inference_dropout_1_layer_call_and_return_conditional_losses_8778n<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
C__inference_dropout_1_layer_call_and_return_conditional_losses_8783n<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
(__inference_dropout_1_layer_call_fn_8788a<?9
2?/
)?&
inputs??????????
p
? "!????????????
(__inference_dropout_1_layer_call_fn_8793a<?9
2?/
)?&
inputs??????????
p 
? "!????????????
A__inference_dropout_layer_call_and_return_conditional_losses_8671n<?9
2?/
)?&
inputs??????????n 
p
? ".?+
$?!
0??????????n 
? ?
A__inference_dropout_layer_call_and_return_conditional_losses_8676n<?9
2?/
)?&
inputs??????????n 
p 
? ".?+
$?!
0??????????n 
? ?
&__inference_dropout_layer_call_fn_8681a<?9
2?/
)?&
inputs??????????n 
p
? "!???????????n ?
&__inference_dropout_layer_call_fn_8686a<?9
2?/
)?&
inputs??????????n 
p 
? "!???????????n ?
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_7366?R?O
H?E
C?@
inputs4????????????????????????????????????
? ".?+
$?!
0??????????????????
? ?
7__inference_global_average_pooling2d_layer_call_fn_7372wR?O
H?E
C?@
inputs4????????????????????????????????????
? "!????????????????????
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_7329?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
.__inference_max_pooling2d_1_layer_call_fn_7335?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_7341?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
.__inference_max_pooling2d_2_layer_call_fn_7347?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_7353?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
.__inference_max_pooling2d_3_layer_call_fn_7359?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_7317?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
,__inference_max_pooling2d_layer_call_fn_7323?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
?__inference_model_layer_call_and_return_conditional_losses_7603y+,-./0123456B??
8?5
+?(
input_3???????????
p

 
? "%?"
?
0?????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_7644y+,-./0123456B??
8?5
+?(
input_3???????????
p 

 
? "%?"
?
0?????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_8528x+,-./0123456A?>
7?4
*?'
inputs???????????
p

 
? "%?"
?
0?????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_8581x+,-./0123456A?>
7?4
*?'
inputs???????????
p 

 
? "%?"
?
0?????????
? ?
$__inference_model_layer_call_fn_7715l+,-./0123456B??
8?5
+?(
input_3???????????
p

 
? "???????????
$__inference_model_layer_call_fn_7785l+,-./0123456B??
8?5
+?(
input_3???????????
p 

 
? "???????????
$__inference_model_layer_call_fn_8610k+,-./0123456A?>
7?4
*?'
inputs???????????
p

 
? "???????????
$__inference_model_layer_call_fn_8639k+,-./0123456A?>
7?4
*?'
inputs???????????
p 

 
? "???????????
M__inference_sig_ver_siamese_cnn_layer_call_and_return_conditional_losses_7896?+,-./0123456??t?q
j?g
]?Z
+?(
input_1???????????
+?(
input_2???????????
p

 
? "%?"
?
0?????????
? ?
M__inference_sig_ver_siamese_cnn_layer_call_and_return_conditional_losses_7948?+,-./0123456??t?q
j?g
]?Z
+?(
input_1???????????
+?(
input_2???????????
p 

 
? "%?"
?
0?????????
? ?
M__inference_sig_ver_siamese_cnn_layer_call_and_return_conditional_losses_8293?+,-./0123456??v?s
l?i
_?\
,?)
inputs/0???????????
,?)
inputs/1???????????
p

 
? "%?"
?
0?????????
? ?
M__inference_sig_ver_siamese_cnn_layer_call_and_return_conditional_losses_8393?+,-./0123456??v?s
l?i
_?\
,?)
inputs/0???????????
,?)
inputs/1???????????
p 

 
? "%?"
?
0?????????
? ?
2__inference_sig_ver_siamese_cnn_layer_call_fn_8035?+,-./0123456??t?q
j?g
]?Z
+?(
input_1???????????
+?(
input_2???????????
p

 
? "???????????
2__inference_sig_ver_siamese_cnn_layer_call_fn_8121?+,-./0123456??t?q
j?g
]?Z
+?(
input_1???????????
+?(
input_2???????????
p 

 
? "???????????
2__inference_sig_ver_siamese_cnn_layer_call_fn_8427?+,-./0123456??v?s
l?i
_?\
,?)
inputs/0???????????
,?)
inputs/1???????????
p

 
? "???????????
2__inference_sig_ver_siamese_cnn_layer_call_fn_8461?+,-./0123456??v?s
l?i
_?\
,?)
inputs/0???????????
,?)
inputs/1???????????
p 

 
? "???????????
"__inference_signature_wrapper_8165?+,-./0123456??}?z
? 
s?p
6
input_1+?(
input_1???????????
6
input_2+?(
input_2???????????";?8
6
tf.math.sqrt&?#
tf.math.sqrt?????????