	??9?eǥ@??9?eǥ@!??9?eǥ@	i̳??8??i̳??8??!i̳??8??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:??9?eǥ@?k*?8@A1??????@YA??ǘ[#@rEagerKernelExecute 0*	gfffffY@2F
Iterator::Model??MbX??!???p8\H@)X9??v???1a0?>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatlxz?,C??!K?R?T*;@)??@??ǘ?14?F??7@:Preprocessing2U
Iterator::Model::ParallelMapV2?l??????!?V??j52@)?l??????1?V??j52@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?St$????!?j?Z?V0@)M?O???19???#@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice9??v??z?!?r?\.?@)9??v??z?1?r?\.?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip[B>?٬??!y<?ǣI@)?~j?t?x?1???|>?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor_?Q?k?!?X,??
@)_?Q?k?1?X,??
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?l??????!?V??j52@)ŏ1w-!_?1|?^?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9i̳??8??I4LOx??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?k*?8@?k*?8@!?k*?8@      ??!       "      ??!       *      ??!       2	1??????@1??????@!1??????@:      ??!       B      ??!       J	A??ǘ[#@A??ǘ[#@!A??ǘ[#@R      ??!       Z	A??ǘ[#@A??ǘ[#@!A??ǘ[#@b      ??!       JCPU_ONLYYi̳??8??b q4LOx??X@