?	??ْy0?@??ْy0?@!??ْy0?@	??=??6@??=??6@!??=??6@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:??ْy0?@?(ϼ?O@A???ܣ8?@Yzq??=R?@rEagerKernelExecute 0*	?O?d?9A2U
Iterator::Model::ParallelMapV2??-*p?@!???,?H@)??-*p?@1???,?H@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate p???o?@!h????H@)?h??o?@1w:	??H@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat9`W?????!@GB7]8??)ڌ?U???1???nߍ??:Preprocessing2F
Iterator::Modeld\qqTp?@!D?T?H@);?Y??!??1??*??S?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip8??9?q?@!??}?? I@)??`??1?????H?:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?? %̄?!?$????C?)?? %̄?1?$????C?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?kzPP?v?!jhQ	?O5?)?kzPP?v?1jhQ	?O5?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?\T?o?@!??????H@)]S ???m?1???lDG,?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 22.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9??=??6@I?]???[S@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?(ϼ?O@?(ϼ?O@!?(ϼ?O@      ??!       "      ??!       *      ??!       2	???ܣ8?@???ܣ8?@!???ܣ8?@:      ??!       B      ??!       J	zq??=R?@zq??=R?@!zq??=R?@R      ??!       Z	zq??=R?@zq??=R?@!zq??=R?@b      ??!       JCPU_ONLYY??=??6@b q?]???[S@Y      Y@q???@-???"?
host?Your program is HIGHLY input-bound because 22.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 