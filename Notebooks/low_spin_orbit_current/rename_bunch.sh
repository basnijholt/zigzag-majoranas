prefix=""
for f in learner_data_\<KB_c0a2a5_\>\<PRMS_84abf4_\>\<PARS_382836_\>\<METR_29c1a_\>\<DIM_*;
do
	common=${f:0:-4}
	echo $common
	prefix=${f:0:13}
	echo $prefix
	
	kb_hash=${f:13:12}
	echo $kb_hash

	prms_hash=${f:25:14}
	echo $prms_hash
	
	pars_hash=${f:39:14}
	echo $pars_hash
	
	metr_hash=${f:53:13}
	echo $metr_hash
		
	dim_hash=${f:66:13}
	echo $dim_hash
	

	suffix=${f:79:4}
	echo $suffix

	echo $prefix$kb_hash$prms_hash$pars_hash$metr_hash"<DIM_8d79d7_>"$suffix
	echo $f
	mv -v $f $prefix$kb_hash$prms_hash$pars_hash$metr_hash"<DIM_8d79d7_>"$suffix
done;
