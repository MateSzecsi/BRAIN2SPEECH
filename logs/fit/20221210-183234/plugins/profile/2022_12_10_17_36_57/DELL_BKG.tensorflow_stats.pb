"?G
uHostFlushSummaryWriter"FlushSummaryWriter(133333??@933333??@A33333??@I33333??@a?٫D??i?٫D???Unknown?
BHostIDLE"IDLE1fffff.?@Afffff.?@a??>????iI???$|???Unknown
uHost_FusedMatMul"sequential_11/dense_22/Relu(1     @b@9     @b@A     @b@I     @b@a?0v???i9????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(133333?_@933333?_@A33333?_@I33333?_@a}$????i?Y	?B???Unknown
?HostMatMul"2gradient_tape/sequential_11/dense_22/MatMul/MatMul(1?????,]@9?????,]@A?????,]@I?????,]@aM-r?2??i??!s?????Unknown
^HostGatherV2"GatherV2(1     ?V@9     ?V@A     ?V@I     ?V@a??ʞ??iw????????Unknown
?HostMatMul"4gradient_tape/sequential_11/dense_23/MatMul/MatMul_1(1      R@9      R@A      R@I      R@a????(z?i???L???Unknown
`HostGatherV2"
GatherV2_1(1fffff?N@9fffff?N@Afffff?N@Ifffff?N@a??c??Dv?i????:???Unknown
?	HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1?????I@9?????I@A?????I@I?????I@a	p
W?<r?i????O_???Unknown
?
HostMatMul"2gradient_tape/sequential_11/dense_23/MatMul/MatMul(1fffff?E@9fffff?E@Afffff?E@Ifffff?E@aҠZI??o?i%9?"???Unknown
cHostDataset"Iterator::Root(1     ?P@9     ?P@A??????D@I??????D@a"?nxF?m?i??????Unknown
?HostReluGrad"-gradient_tape/sequential_11/dense_22/ReluGrad(133333?B@933333?B@A33333?B@I33333?B@a??~ x?k?i'???????Unknown
xHost_FusedMatMul"sequential_11/dense_23/BiasAdd(1     ?A@9     ?A@A     ?A@I     ?A@a????i?i"?>?f????Unknown
?HostSquaredDifference"$mean_squared_error/SquaredDifference(1      A@9      A@A      A@I      A@a?o??h?i??%?????Unknown
?HostDataset"4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat(1      ?@9      ?@Affffff;@Iffffff;@a?^=7s?c?i?:]'????Unknown
rHostDataset"Iterator::Root::ParallelMapV2(1??????9@9??????9@A??????9@I??????9@a???ؾb?i?I; ????Unknown
iHostWriteSummary"WriteSummary(13333335@93333335@A3333335@I3333335@a??Q;z?^?i??X=)!???Unknown?
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1fffff?2@9fffff?2@Afffff?2@Ifffff?2@a??z?v[?i?h???.???Unknown
uHostReadVariableOp"div_no_nan/ReadVariableOp(1??????2@9??????2@A??????2@I??????2@a??z?D[?i?%?Nh<???Unknown
VHostMean"Mean(133333?0@933333?0@A33333?0@I33333?0@a??vDX?i??削H???Unknown
THostSub"sub(1??????0@9??????0@A??????0@I??????0@a(??BX?i?C+?T???Unknown
?HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1333333,@9333333,@A333333,@I333333,@a???c@}T?i`)u??^???Unknown
?HostBiasAddGrad"8gradient_tape/sequential_11/dense_22/BiasAdd/BiasAddGrad(1      *@9      *@A      *@I      *@a??4)?R?i?É?Jh???Unknown
?HostDataset">Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1??????2@9??????2@A??????&@I??????&@aT??+?kP?i?????p???Unknown
wHostDataset""Iterator::Root::ParallelMapV2::Zip(1ffffffP@9ffffffP@Affffff$@Iffffff$@a-S#?ߤM?i]%???w???Unknown
iHostMean"mean_squared_error/Mean(1??????#@9??????#@A??????#@I??????#@aRg@??L?iw??F???Unknown
[HostAddV2"Adam/add(1??????"@9??????"@A??????"@I??????"@a?3?/?QK?i?k??????Unknown
gHostStridedSlice"strided_slice(1ffffff!@9ffffff!@Affffff!@Iffffff!@aⷴ?HI?i2԰?A????Unknown
?HostBiasAddGrad"8gradient_tape/sequential_11/dense_23/BiasAdd/BiasAddGrad(1333333!@9333333!@A333333!@I333333!@a?i}v?H?iw.P??????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1?????? @9?????? @A?????? @I?????? @a(??BH?i??V?????Unknown
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1       @9       @A       @I       @a8<??@G?iи?ZY????Unknown
l HostIteratorGetNext"IteratorGetNext(1??????@9??????@A??????@I??????@a?d?q??E?i)0?Ĺ????Unknown
?!HostDataset"NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1333333@9333333@A333333@I333333@a????@7E?i?Д????Unknown
u"HostSum"$mean_squared_error/weighted_loss/Sum(1??????@9??????@A??????@I??????@a?0c??D?ie?p?
????Unknown
V#HostSum"Sum_2(1??????@9??????@A??????@I??????@a???U?xC?i_ ?4?????Unknown
[$HostPow"
Adam/Pow_1(1      @9      @A      @I      @a??4)?B?i?m?7?????Unknown
?%HostDataset".Iterator::Root::ParallelMapV2::Zip[0]::FlatMap(1ffffff8@9ffffff8@A333333@I333333@aB%#>?@?iP?Sٻ???Unknown
w&HostMul"&gradient_tape/mean_squared_error/mul_1(1333333@9333333@A333333@I333333@aB%#>?@?i???????Unknown
?'HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1333333@9333333@A333333@I333333@aB%#>?@?i?GZ?F????Unknown
u(HostSub"$gradient_tape/mean_squared_error/sub(1??????@9??????@A??????@I??????@aN??vא@?i??7?j????Unknown
e)Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a?r?????iЍ??i????Unknown?
v*HostAssignAddVariableOp"AssignAddVariableOp_2(1??????@9??????@A??????@I??????@a?*?gGc??i???GV????Unknown
}+HostRealDiv"(gradient_tape/mean_squared_error/truediv(1      @9      @A      @I      @aF??=?iv<J?????Unknown
?,HostDataset"@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1??????@9????????A??????@I????????a?FE??4?i:?V??????Unknown
u-HostMul"$gradient_tape/mean_squared_error/Mul(1??????@9??????@A??????@I??????@a?FE??4?i????3????Unknown
Y.HostPow"Adam/Pow(1ffffff
@9ffffff
@Affffff
@Iffffff
@a?D??r.3?i~?N?????Unknown
T/HostAbs"Abs(1??????	@9??????	@A??????	@I??????	@a??蒥?2?i'۩??????Unknown
t0HostReadVariableOp"Adam/Cast/ReadVariableOp(1333333@9333333@A333333@I333333@aB%#>?0?i?k?????Unknown
|1HostDivNoNan"&mean_squared_error/weighted_loss/value(1333333@9333333@A333333@I333333@aB%#>?0?i?#-S#????Unknown
?2HostReadVariableOp",sequential_11/dense_22/MatMul/ReadVariableOp(1333333@9333333@A333333@I333333@aB%#>?0?iV???>????Unknown
3HostFloorDiv")gradient_tape/mean_squared_error/floordiv(1333333@9333333@A333333@I333333@av{]\x?+?i.?t"?????Unknown
T4HostMul"Mul(1ffffff@9ffffff@Affffff@Iffffff@a??.޼*?i?T??????Unknown
]5HostCast"Adam/Cast_1(1??????@9??????@A??????@I??????@a?[ ?C?)?i#a?$B????Unknown
t6HostAssignAddVariableOp"AssignAddVariableOp(1??????@9??????@A??????@I??????@a?[ ?C?)?i)?X?????Unknown
V7HostCast"Cast(1??????@9??????@A??????@I??????@a?[ ?C?)?i/??t????Unknown
}8HostMaximum"(gradient_tape/mean_squared_error/Maximum(1??????@9??????@A??????@I??????@a?[ ?C?)?i5A??????Unknown
v9HostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @a8<??@'?ii?.????Unknown
~:HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1ffffff??9ffffff??Affffff??Iffffff??ah?t?t&?i??x)?????Unknown
u;HostSum"$gradient_tape/mean_squared_error/Sum(1ffffff??9ffffff??Affffff??Iffffff??ah?t?t&?i??D????Unknown
b<HostDivNoNan"div_no_nan_1(1333333??9333333??A333333??I333333??aɌ???#?ixK?Ā????Unknown
?=HostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1333333??9333333??A333333??I333333??aɌ???#?i????????Unknown
v>HostAssignAddVariableOp"AssignAddVariableOp_3(1ffffff??9ffffff??Affffff??Iffffff??aZ݋?pF ?i??_?????Unknown
X?HostCast"Cast_1(1ffffff??9ffffff??Affffff??Iffffff??aZ݋?pF ?im???????Unknown
w@HostReadVariableOp"div_no_nan_1/ReadVariableOp(1????????9????????A????????I????????a???9?iB?D??????Unknown
wAHostCast"%gradient_tape/mean_squared_error/Cast(1????????9????????A????????I????????a???9?i
?a?????Unknown
?BHostReadVariableOp"-sequential_11/dense_22/BiasAdd/ReadVariableOp(1????????9????????A????????I????????a???9?i?/?????Unknown
`CHostDivNoNan"
div_no_nan(1????????9????????A????????I????????a?[ ?C??i??2?g????Unknown
vDHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1      ??9      ??A      ??I      ??a8<??@?i	???!????Unknown
?EHostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1      ??9      ??A      ??I      ??a8<??@?i#J"??????Unknown
vFHostAssignAddVariableOp"AssignAddVariableOp_1(1????????9????????A????????I????????a?FE???iTt?0?????Unknown
aGHostIdentity"Identity(1????????9????????A????????I????????a?FE???i??Ɨ*????Unknown?
vHHostReadVariableOp"Adam/Cast_2/ReadVariableOp(1????????9????????A????????I????????a??蒥??i?5?d?????Unknown
yIHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1ffffff??9ffffff??Affffff??Iffffff??aZ݋?pF?i,:z?A????Unknown
oJHostReadVariableOp"Adam/ReadVariableOp(1333333??9333333??A333333??I333333??av{]\x??i??[2?????Unknown
wKHostReadVariableOp"div_no_nan/ReadVariableOp_1(1333333??9333333??A333333??I333333??av{]\x??i=? ????Unknown
?LHostReadVariableOp"-sequential_11/dense_23/BiasAdd/ReadVariableOp(1333333??9333333??A333333??I333333??av{]\x??i??f?????Unknown
?MHostReadVariableOp",sequential_11/dense_23/MatMul/ReadVariableOp(1333333??9333333??A333333??I333333??av{]\x??i     ???Unknown*?G
uHostFlushSummaryWriter"FlushSummaryWriter(133333??@933333??@A33333??@I33333??@a?{}??:??i?{}??:???Unknown?
uHost_FusedMatMul"sequential_11/dense_22/Relu(1     @b@9     @b@A     @b@I     @b@a?#????i??????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(133333?_@933333?_@A33333?_@I33333?_@a?s?9?a??i?f??=H???Unknown
?HostMatMul"2gradient_tape/sequential_11/dense_22/MatMul/MatMul(1?????,]@9?????,]@A?????,]@I?????,]@a{^r??Č?iK0?!Q????Unknown
^HostGatherV2"GatherV2(1     ?V@9     ?V@A     ?V@I     ?V@a?<?q???i? ??????Unknown
?HostMatMul"4gradient_tape/sequential_11/dense_23/MatMul/MatMul_1(1      R@9      R@A      R@I      R@a`?v;ӿ??i???5?\???Unknown
`HostGatherV2"
GatherV2_1(1fffff?N@9fffff?N@Afffff?N@Ifffff?N@a?!?9~?i4??u?????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1?????I@9?????I@A?????I@I?????I@an???x?i6ˮ|????Unknown
?	HostMatMul"2gradient_tape/sequential_11/dense_23/MatMul/MatMul(1fffff?E@9fffff?E@Afffff?E@Ifffff?E@a?]?R?u?i???R?????Unknown
c
HostDataset"Iterator::Root(1     ?P@9     ?P@A??????D@I??????D@anǽ?'Pt?i[t??M???Unknown
?HostReluGrad"-gradient_tape/sequential_11/dense_22/ReluGrad(133333?B@933333?B@A33333?B@I33333?B@a%??[??r?i	?7??C???Unknown
xHost_FusedMatMul"sequential_11/dense_23/BiasAdd(1     ?A@9     ?A@A     ?A@I     ?A@a?h??q?i+?	X?f???Unknown
?HostSquaredDifference"$mean_squared_error/SquaredDifference(1      A@9      A@A      A@I      A@aZu?c?p?i??5????Unknown
?HostDataset"4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat(1      ?@9      ?@Affffff;@Iffffff;@a?))*?k?i@??9????Unknown
rHostDataset"Iterator::Root::ParallelMapV2(1??????9@9??????9@A??????9@I??????9@a?!L??pi?ib>???????Unknown
iHostWriteSummary"WriteSummary(13333335@93333335@A3333335@I3333335@aq?0Ý?d?io?D?????Unknown?
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1fffff?2@9fffff?2@Afffff?2@Ifffff?2@a???1?b?i??H5????Unknown
uHostReadVariableOp"div_no_nan/ReadVariableOp(1??????2@9??????2@A??????2@I??????2@ac}?4IWb?ix?%??????Unknown
VHostMean"Mean(133333?0@933333?0@A33333?0@I33333?0@a?S??w`?i|N;???Unknown
THostSub"sub(1??????0@9??????0@A??????0@I??????0@aY3?j^`?i?c??b???Unknown
?HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1333333,@9333333,@A333333,@I333333,@a??̨?[?i?o ?I%???Unknown
?HostBiasAddGrad"8gradient_tape/sequential_11/dense_22/BiasAdd/BiasAddGrad(1      *@9      *@A      *@I      *@a??ǎM?Y?ig???2???Unknown
?HostDataset">Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1??????2@9??????2@A??????&@I??????&@ay?^IV?ip$@=???Unknown
wHostDataset""Iterator::Root::ParallelMapV2::Zip(1ffffffP@9ffffffP@Affffff$@Iffffff$@al&B!?T?i????NG???Unknown
iHostMean"mean_squared_error/Mean(1??????#@9??????#@A??????#@I??????#@ajC?'5?S?i%?;Q???Unknown
[HostAddV2"Adam/add(1??????"@9??????"@A??????"@I??????"@ade?ŉR?i?W*?VZ???Unknown
gHostStridedSlice"strided_slice(1ffffff!@9ffffff!@Affffff!@Iffffff!@a\?B](Q?i?Y?%?b???Unknown
?HostBiasAddGrad"8gradient_tape/sequential_11/dense_23/BiasAdd/BiasAddGrad(1333333!@9333333!@A333333!@I333333!@a[????P?i?fk???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1?????? @9?????? @A?????? @I?????? @aY3?j^P?i5(hK?s???Unknown
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1       @9       @A       @I       @a??DM??O?i]y??x{???Unknown
lHostIteratorGetNext"IteratorGetNext(1??????@9??????@A??????@I??????@a?yg0M?i?W?Ă???Unknown
? HostDataset"NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1333333@9333333@A333333@I333333@a?ҁ?L?i???????Unknown
u!HostSum"$mean_squared_error/weighted_loss/Sum(1??????@9??????@A??????@I??????@a?ʤ?27K?iJ??\Ő???Unknown
V"HostSum"Sum_2(1??????@9??????@A??????@I??????@a?F?0@mJ?i??۬`????Unknown
[#HostPow"
Adam/Pow_1(1      @9      @A      @I      @a??ǎM?I?ì??ɝ???Unknown
?$HostDataset".Iterator::Root::ParallelMapV2::Zip[0]::FlatMap(1ffffff8@9ffffff8@A333333@I333333@a{?X|?F?i
?U??????Unknown
w%HostMul"&gradient_tape/mean_squared_error/mul_1(1333333@9333333@A333333@I333333@a{?X|?F?iG?k?9????Unknown
?&HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1333333@9333333@A333333@I333333@a{?X|?F?i?????????Unknown
u'HostSub"$gradient_tape/mean_squared_error/sub(1??????@9??????@A??????@I??????@az??{F?i?GC??????Unknown
e(Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @au.e??E?i??\"?????Unknown?
v)HostAssignAddVariableOp"AssignAddVariableOp_2(1??????@9??????@A??????@I??????@as?'?LE?i??!HP????Unknown
}*HostRealDiv"(gradient_tape/mean_squared_error/truediv(1      @9      @A      @I      @aj?Jб?C?iq??t>????Unknown
?+HostDataset"@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1??????@9????????A??????@I????????a????f<?i?]n8?????Unknown
u,HostMul"$gradient_tape/mean_squared_error/Mul(1??????@9??????@A??????@I??????@a????f<?iG?W????Unknown
Y-HostPow"Adam/Pow(1ffffff
@9ffffff
@Affffff
@Iffffff
@a???F:?i?#?????Unknown
T.HostAbs"Abs(1??????	@9??????	@A??????	@I??????	@a???=T>9?i????????Unknown
t/HostReadVariableOp"Adam/Cast/ReadVariableOp(1333333@9333333@A333333@I333333@a{?X|?6?i??5ߜ????Unknown
|0HostDivNoNan"&mean_squared_error/weighted_loss/value(1333333@9333333@A333333@I333333@a{?X|?6?iD???x????Unknown
?1HostReadVariableOp",sequential_11/dense_22/MatMul/ReadVariableOp(1333333@9333333@A333333@I333333@a{?X|?6?i??K?T????Unknown
2HostFloorDiv")gradient_tape/mean_squared_error/floordiv(1333333@9333333@A333333@I333333@af`\.??2?io?1ֲ????Unknown
T3HostMul"Mul(1ffffff@9ffffff@Affffff@Iffffff@ab?m??$2?i+?o?????Unknown
]4HostCast"Adam/Cast_1(1??????@9??????@A??????@I??????@a^X??Z1?il ?"????Unknown
t5HostAssignAddVariableOp"AssignAddVariableOp(1??????@9??????@A??????@I??????@a^X??Z1?i?=&N????Unknown
V6HostCast"Cast(1??????@9??????@A??????@I??????@a^X??Z1?i?{?y????Unknown
}7HostMaximum"(gradient_tape/mean_squared_error/Maximum(1??????@9??????@A??????@I??????@a^X??Z1?i?[?ܤ????Unknown
v8HostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @a??DM??/?i!0M??????Unknown
~9HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1ffffff??9ffffff??Affffff??Iffffff??a??g	?-?i?ƍ[}????Unknown
u:HostSum"$gradient_tape/mean_squared_error/Sum(1ffffff??9ffffff??Affffff??Iffffff??a??g	?-?i]??\????Unknown
b;HostDivNoNan"div_no_nan_1(1333333??9333333??A333333??I333333??a????9?*?i?wf
????Unknown
?<HostCast"2mean_squared_error/weighted_loss/num_elements/Cast(1333333??9333333??A333333??I333333??a????9?*?iǒ?B?????Unknown
v=HostAssignAddVariableOp"AssignAddVariableOp_3(1ffffff??9ffffff??Affffff??Iffffff??awp??&?i.???????Unknown
X>HostCast"Cast_1(1ffffff??9ffffff??Affffff??Iffffff??awp??&?i?U5z????Unknown
w?HostReadVariableOp"div_no_nan_1/ReadVariableOp(1????????9????????A????????I????????aoh9r??$?i,y|>?????Unknown
w@HostCast"%gradient_tape/mean_squared_error/Cast(1????????9????????A????????I????????aoh9r??$?iÜ?h
????Unknown
?AHostReadVariableOp"-sequential_11/dense_22/BiasAdd/ReadVariableOp(1????????9????????A????????I????????aoh9r??$?iZ?
?R????Unknown
`BHostDivNoNan"
div_no_nan(1????????9????????A????????I????????a^X??Z!?iPh?@h????Unknown
vCHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1      ??9      ??A      ??I      ??a??DM???iu???d????Unknown
?DHostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1      ??9      ??A      ??I      ??a??DM???i?<>a????Unknown
vEHostAssignAddVariableOp"AssignAddVariableOp_1(1????????9????????A????????I????????a????f?i?h4PD????Unknown
aFHostIdentity"Identity(1????????9????????A????????I????????a????f?iD?*?'????Unknown?
vGHostReadVariableOp"Adam/Cast_2/ReadVariableOp(1????????9????????A????????I????????a???=T>?iȃ?s?????Unknown
yHHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1ffffff??9ffffff??Affffff??Iffffff??awp???i|4(?????Unknown
oIHostReadVariableOp"Adam/ReadVariableOp(1333333??9333333??A333333??I333333??af`\.???i_??9????Unknown
wJHostReadVariableOp"div_no_nan/ReadVariableOp_1(1333333??9333333??A333333??I333333??af`\.???iB?????Unknown
?KHostReadVariableOp"-sequential_11/dense_23/BiasAdd/ReadVariableOp(1333333??9333333??A333333??I333333??af`\.???i%??h????Unknown
?LHostReadVariableOp",sequential_11/dense_23/MatMul/ReadVariableOp(1333333??9333333??A333333??I333333??af`\.???i     ???Unknown2CPU