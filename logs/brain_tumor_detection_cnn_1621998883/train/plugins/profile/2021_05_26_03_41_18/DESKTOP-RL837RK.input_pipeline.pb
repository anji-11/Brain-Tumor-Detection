	?xwd$I?@?xwd$I?@!?xwd$I?@	?*?|?a??*?|?a?!?*?|?a?"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?xwd$I?@?j???#??A.??e?H?@Y?%???o??rEagerKernelExecute 0*	/?$?o@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateg?????!%?R${Q@)l
dv???1??KV?P@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??fF???!?2???0@)?????A??1??
)?-@:Preprocessing2U
Iterator::Model::ParallelMapV2?ܚt["??!t?? ?@)?ܚt["??1t?? ?@:Preprocessing2F
Iterator::Model?g\8???!o??#? @)0?????1?X????@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip2v?Kp???!?????V@)[A?+???14˾??
@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?????
??!?? 
@)?????
??1?? 
@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor7ݲC??v?!?]?s@)7ݲC??v?1?]?s@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???ȭI??!?I??$?Q@)o??m?n?1Z?? ???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?*?|?a?I|ٻ??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?j???#???j???#??!?j???#??      ??!       "      ??!       *      ??!       2	.??e?H?@.??e?H?@!.??e?H?@:      ??!       B      ??!       J	?%???o???%???o??!?%???o??R      ??!       Z	?%???o???%???o??!?%???o??b      ??!       JCPU_ONLYY?*?|?a?b q|ٻ??X@