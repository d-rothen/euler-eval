"""Bundled NIQE and FADE natural-scene reference model parameters.

The values are losslessly compressed float64 arrays from the official LIVE
software releases. Keeping them in the package makes both metrics deterministic
and usable offline. See ``THIRD_PARTY_NOTICES.md`` for provenance and terms.
"""

from __future__ import annotations

import base64
import zlib

import numpy as np

# Uncompressed SHA-256: f6284a2b53fd40bbc960472b76ec55b67a299a0270294b804c847e99b8cc1e30
_NIQE_DATA = (
    "c-"
    "lRh_e0O^8^uG9P*#e^mZGepK~`5&2}yhJO;dY*+NDK9Wfe(^LJ3iBC0b@=6ea4RqEsSOzJB<g{)Ov(U*|gKb>4F&"
    "f5Zhn=9U-NTeKtM_h;&?3_7!Gq>C0`iglA-"
    "d7T{h_Iv)RtR#!=R>g|u^|blWSkstvH@)X{R*n3+MA_;4Mr5<^((I@F<}P#UXxT3_18<LRGOS%59xr^1l{%}J$_U"
    "reXzhPLMP=`iZsljYbc=2ZzHqEvYB-"
    "a<Ns1Zs_)$ac%XBW;F1ttPBu@xNRr1K|zw>!;psJUea(e|ntm~oW7t@~e%`2sM8!v8^I9E+lt`!^i)<2+0%lLSM1"
    "AEA5=+m@6QWweNrN&clxBE16)Aq5J`UWx*3H&|dQx9dF`&zv@B%5s)U-"
    "fi#S_5r!H(6pbua@4WZ;e{q*h8g$R)3}>WV8J%+dHcQ8%RUF`H=3X8tScc;;#GWEvbKS_$_ek28n;kd^jvoO5Z$J"
    "M0nXIk+#WO6VHcd=%Ub)cax<r)5f+-"
    "9%CZc$X!tVR`n@2w#&G3@#RbTw8u}+Ex4(GR*5^DnkaXTLf>r6K9wEIOg|c(x>#RG#v3A*`K-+&uA-"
    "d1`Tvzt|Hg+$<EBS2&ifTD>thOuE6b+R=<GSVwY%~_-"
    "R(va@VRnIf8;XBR=ef4)E7`;Q=4htcNZ!gnYdI%;50@2bnRK1kW1J8tV-c-"
    "FQ7{98}|2id9a`2)`C@rIb;*97cA?QL!V+6nM=wR(1(t*{AyF9m~HE$#M^@B=xE3XojF&sNO#wCM-Pj95{sXe(aI"
    "mlD)~KXE^o-Cg@wyy)%CN-"
    "=Q8iRr;!F%4!K+Y5&b#)D2dKKdbPUDn)u(OPIhoeAn}Y($3h>akiCahK%(7IvVA|Oee<d&iyY!_apOKhGD8mAn!X"
    "&N!sUg=W*upCbj`{n%`IMR>WAwW&YejirzfhykJcs9a@DYj2|j65Uo)+l)_bzH)buXPoD|yFtM$CwE`ct)YBEmZI"
    "Wjc8blfyBgHpqSgUUT(NmZHh$}$z$gNbEUdCt*vpd$KtpmHo7FKioq@G*w&Y>bZoU|_*!jJP{`{D>j>&??7wN2BT"
    "S?4dBHz$mJkRr*{!#D%%H%8iYT93o?lW0pq`htuxYFXQ7PBS~Snb#h~n6T47qt<7V8h~y`H65f9<eEhv&F4sdh?0"
    "zbgOnpn#bAy5%X#1)ng$2=}wCrq$ik4VB<p+rrSZ+(EpvNPgI$1B|_+!eyU9XCyb$lJo32S1gL_YY8VEPdX8$3M4"
    "<*g-q=#!i+#v4m9b5#DDZyZ4-"
    "(*;De_8p<&rno8fH!N7)zb=Qn_+!bNPvyqK;&2*qT%kK9CZ7&}T>PKZuS7a1qo`Rq5=GIM<!=RZZK6x=CGGP+g-"
    "}k$!X|^M;Z!j2d{=E<6b-MC+iCB(jR~Ib<of-PL$-"
    "}aH%0V>(wn1ul=<Q!s5dsJQYFWX@$*s(@7XY7V=OtNDv%~d7OM9*gp<@a!%bVC8L{Yx(@J^vhtW>uV>0*M0>}3V?"
    "*(&t;wZN{Rn(34NGevJ4UuQomL8+WogB%4FSSLi(~TC|<~|o#XUGoN2DeGwx27S-"
    "QNLa;ExP(%b@iRs2T0&}q1`nfK5EuI)zMI7LNfzKSLV&ur?DK#xB8;aBrMs~)8#D4wwF!sT6*1-"
    "W~)4TXyK+y1~G?zlrM^+TKP`Jx@C@(I%VwS5U(Z4ANO36zQvaA$@rTHjIAeW55d6wmbT=gpl~Qu%!%A@I>vSGG-g"
    "Q`4C5LzY^Z*ndv$Q84(a(Xekx&ZN8Xm&uhh24v5R-"
    "E<W%We&_ISO*GdI_O3S*jsOXC=8LOQz)fSUyStbiDbgeAtz=wN%(_8e$^#yx`_ky|niuay&^JV~j{$gG?_Boh}&T"
    "f7@>mr91B-"
    "bcZ6iZV4&b$X(D*Vag_Liebmx5`lxpc;z!4T@cnD+HqngvT3N{qWQfkO*(w>VEs@ui6&_092XLdfZQ^6evXmTcg$"
    "ME&%CIrP0ZylO$M4{bNJ)Z&YZqbY@*u0g}0^dL&v^M=<!dThd#y(CVL1@h=bT`uvYk@D@5dH;A&&s^(8Pk+17iQ{"
    "UcJ0t~}noItR`4#ST_Pj$!t?)sr-JO`$qVG(f8j>%Z)Y4|xv@^4eo*krj$v<aENV-"
    "s^r=G;n!UH5ClH~L(Nt?+p={s?&<{;f~I4>-"
    "G&v{&Tp)c4Yycf)+*}p`tS0R?v`Ulr%t0mAJ|3Fz^dtLT)P204`Ya+=fRxVyFGJ-TjoUE6+$5EKG=1aLW8)j88*}"
    "Cmp6lKJAEh)MZN~*!)<sI|l$Ry>d=}d1scBMmL8vmYXD(>jd6#T%Ud3H^~R+^_sZfq*=n9CvBRPu^U^f)wJJ=6T)"
    "E)|l?7F?Qp+mDprm5)5R7edafBJ_psacI%;?UK3AjhOsdeyf8|f=Ot#kW_zs0KL$wi16GRL=V#Uj{5F1XVW&?76`"
    "|Mkfrv`fz8T3WXs~q^dtl6`X}oX&D`c}ZJnp)FS`)px!5G7^wWF%9zl1ZFW4iz7tH0bTk(Vq`6<-"
    "hGKFSL)nb>$R;)gACz>Xj{wmqKH-@+`8JYN4B~#wx<Rac`YZhHx_m*!kmeMz{m5x^;sM-C_(t}HrsY+w!B5pq`HW"
    ">ZwiO)ZWNHf^CBKuo7aXHxiNWES_M(gwW&WR?G(@zu64~wHo_jEs3*M0*!(D!{fJ2i+NmGaijeHKoa)F!Uuv5TRV"
    "T&)cSQw^D8&TG*G#V{(bnm4k_Ka_f#dPFBCMUu2k+kCZd3l<mQP$2zpICU*?Z4}50q~02p$s*4qsJce>&p%&`nA+"
    "S~MWW}z$jV7f^zG}w@jH!s1l@(cV2|)#Fqg?zzxWb8Tq$#sul*BMTlQ~7jJt$}3Y~6U^t|t#5*t1?fA6Ti9?dE<4"
    "(2Y`VOJ;4G}@^m%#tGCjO^o9VaC~!mm3-l$a2b~vwXSwEUqB_MMXOo-Bp|u<|n+D3VZ$>_@Qb{{3B|Q!msS1Ph-"
    "bA-{cywC-EP{_3aGl%lUW>ZYe49lUWeF>d^$&x2D5vim)xEYWqcW%IskCY-4hl4I5F>sW-"
    "no%hpotx%|P^n>W*Lds*(L&g~Sll)v!)vfcDM;rwqw6A?PUJ4joBR+G1rM9G=?iZoDjRogE>mEd1+r*V&<yU-Wx5"
    "#9^ta;nRIy6J;(^6RqwC=<k?=d#;f<-"
    ";OLc0x!|=|+2IoNLKju>9}$F|(>_nHog@if=k2BppeMN;(}hW;wAg&H06%*%4&6CtiQk&H(Zf@3WoxJ(WDoRxVTj"
    "8cwPX;ymXSeCVERV%x2$%GA|eHGTax53=2&eS7^iAKG$ld(PHWU+OPgbZEwPE4I47bLBN%Ke9ewI=%U=2YF5xSFH"
    "cni`G{!Q8k)k!rD1a{W`OKiDOv2Zq`jVy1-"
    "?yt<lw!wr$wD*6g|w8*G^QPV<HjO%87CJlg6u?(5)RaHnyPpu5l)>=E7z=Aw5yn)_{XI0es<>MdOuO(PA>0|AES?"
    "D2r-?!=sMS`;%U>vL8JO@5W-EpsZ0)TT-"
    "77p^p8tK9;MrH+SFxT03`iMzp+wY4HkZq+gBa+fi!d=*2rO1FdaDuPJt!uvxBdzINUY57pGL;jRG@O?mVe*n?6ir"
    "{%#fuvpev}`0$lzpjLsh{rUNAo2|`u9WzkW&ChVC!99y3VW6^Rv@{t>jGW_E8F?xm%y?mVW0&S9Q4Rhs%7(X8Z2f"
    "8RiBoa@aM^TquzGSG5>)hIozpUHCfq7u;#wBj_&l1$%_|g1KyWf3^2yN(!}=NUzAgV#ho`ENq+iD3*Fc2J6l_M$o"
    "b5e1BE06k2$0YFhmoOLp}pe`$nHEGa39@K5LrC+Gg2oEdK~l4R`H`1&o0q$H7a|9(pp`EBNH(2UigGM;r4Qh0)>H"
    "S~D)uI1rmI`;UPp<6VWM7C)Eysp7ce(8M`-"
    "p3(@EUm^X%R_0DoV==qqv?;emAH4h4I3Nd{;xkioLpz7*Ew(pQq%sIZf4t~h^L09SAV)8<M#cT?UouwV|UN$DBTY"
    "n&jIke@OAJnxYM{t&|T;Y_6Y9<b17Dli_sczpjOYj3X{2A*tIuXPYp)tu)@KXfb(Tm%uY@BuT4>=3i;uK85?X_l4"
    "-4jL7zEmaCIrWHD8&Xma`YJsMKYx#!21s7vz|z^LhGlM~w8c{`}c{MTO0@XnLs~sKWjTovd3&n;EZO<3O9tK1zNs"
    "&?B;5o6X#qwYOnl4WpIS0%iO=*sRbovoG<c^edA4<fU&0Of&zW&5WsYsM2QDrvu$Gj4$n8M*ajv*7~DJ<II9hOyc"
    "+n-a6i0<M{|V0Dc#~4*msq8utjg3w^;J;k{rkZc(2@l#HTj$Xx91UQ-UeiU}N8UK>LrA&yLCq8)2`#1@1(Mp54Vh"
    "oe{RIb^=Rzfa;wCOtb+>lwE`hWds=Pk)&EcYiXIZ+h2GW18uQ#8yfBQh>T%QLI4#9hEGW5>E&qxx&=l|NPQtGt0{"
    "s$+-"
    "rQvibupmo2{Jv~pp|iHCk<+VJE1?fo{)E_C^hr(c7pdE+#bwyOszoX;aKQ`?u?Z}0o(#B*Ji;T%zBW*I;xW#^~LS"
    "$mD=M&u*p0Qg<_I`|jdY1|{|F7yR^g!h8EOq#dbL$M}+rYQ9OO^p#$c<l0Q5qnEE)or0j%AQaXoS_jV=kG_~^k#7"
    "+zo%0{SYgi?i=h9$BsMyY`jYY%!?MA{{Om|c=Rb}rUQ}%H;^TpHK4jGKqvK$?FEuK5y)v;<WUeVY44$p^CJVdgE+"
    "WC+l=J!d?bJ!$wDoHKId#E(ti7bZ_4{ETvRp2k;_}ZyvJ-3-"
    "a*6h$l8@zmJlrNsc>cT}PM>}0nuAJ9N`U)#o<?p&K0*$F--WM(f5Dx`J%a8+U$94bFPKYGTC3Gon-rS$D_FPavMr"
    "mW)aktIZ7glnTyRw^Hk_h6=T+Quzeuu@b^GkkB+v`r@9U&}qDb#*{ak%lU6Nn$RCE4ceeFoAe_eh*jOOhB!K)(_O"
    "_RT#9Dek67uzKMl_K0Y#Fy9TeWaN~viwJPN)|;?i`(OoTPw`j{imgzRf%Er==S{N*~)=*sNvJI?CVivVz_OAe~l%"
    "Z|7dx@^BLh3aUq^Z$tz%-GayeRHzFS)2f**b*TKKwPU9XyccCxXBfJ;P<@fvz1_u%jP#4*>9-"
    "iRN;`H^hgrv0C+e=CLb3)}=XR*KG%%ytl!svkNXMP!Ge>Po!XZkiuS9fRchjy~=Ck{UfvlLig&$F+0)g;;d6L)O#"
    "hSh1-<JBJB$(l^IbB~ByjttB7c=~z%HF?JK&GwAAh&Xfj8Ta6cq7pm%U$1)QWHmNJY}btS#+poEn{DKu-"
    "{zFNB||;%l`(Tma@(f9|MzY17jOpTY2-%aBjf=1UHCfq7u;#wBj_&l1$%_|g1OY6Us7>GH-"
    ";jk+g`8TA3}M}W{P{FGHHD|Pul1A(IgTlu`~bJU!BQR&Ra1m%pTt1+TJAMOK<dbZ7$XN)0{_h%%`pjpkwzNFX`&)"
    "u}5K?R1xM+Pq-"
    ">tdsq6>tiGL{LMDE6?at=GlOGLO_~m%tnvwu|S*P}1TEL6c7Vep{r^t^YY&61oKH0GTh5M^(J_k|MA~)e&$AjZs3"
    "j76}0eKp^5%~x?0Dc#~4*msq8utjg3w^;J;k{rk_b$BBdN1xze4m0gint%8YSCunp}#sfqc5d+({^8q%JE%&v__D"
    "$o(c7=)9@rkgMo_Czxhc}|Buh`Paldh?LBPmpv36oP=|Ge7ik3!Se9S%qQyPGENAcYB0=e&TLu%1*aO3`r0Fw!DR"
    "6p}qnf!py$qi6GFjD=R_6b)6D{4tlz21ZZXNNa(EGc0c5U()=XKyx;4k0|$kWJ;$VbQl@VoGJ@GrR2xJS@k=nM7;"
    "?*(%aZRXN2eBVVn=00!qzZBES5Z??T>+3XH(RNiO#+za#_`{liouzcar}C!l7f3qoWHtYy8??#ovtQ8PT;9m<GjG"
    "D&Tym;tus7(tK;PEw<#m5oM#G0s_3srjVO~j+D|y-"
    "qXm!O_^K)+J>Ggqajn<B3^wrPpP1iafrlc|HnN{pX+Uivt=RNt{IL8F91D67S0cSv-Ms7qtLJol6g|CBu!JWoEg6"
    "=|Jut#_=n9HOq+w3M*X3|TG!?w@c6Y0=`l#uzp%G5V3&@E^mO_p9UK^|doG<OQ$JHMPnx^HZIb?@~RbTIIX5^q&B"
    "O<QwY|Ip1tWa4lmRVh1xHZSFS-"
    "7IXy)c6N(P0ffS!S=SnzLQbZ*I{$oz%!nbR$i{UGuxI$_@9i>sg5JDkYb06Ws&227aSA34qOWS1)KqS8o3eq2sr?"
    "L7rqYu1$P?v2)YY>!5-ngU@nRU&dLd1he`X?m&BVUs??b788TZYl>RMpzL(S-Nf+O#<qN+|q-k$7B-*?8k>-"
    "Hvc~RofviGlw`Gz8>$wRj(EI5IR)}8Zjy|je|YRl}nArwy10}~|N|LURgDSyY2X$d5pC^1v#^Dbte@nwS0vv4x%n"
    "WNUI$r<O~;Je_M;C0|q;4k0|$kWJ;$VbQl@VoGJ@GrR2xJS@k=nM7;?*((o%Z?UPS?ofA%V(c&e`Ul(N-"
    "C~TB101U@S`dBpbgath%8Q#GG!ke+Jn>7^~p!KV{&H1EOs!q;J0k134Pbe+qdcFRF<@D^8<cq1@cRnJBg=IhH__Q"
    "^(;y<Be#Rq^WLo4O27QO=Pn45rdW%sPv&|lkM{xK-"
    "r&37nBaBbQs6J(49L^SjmSsH0r0!<b?`5^)3`^_UFZw;2=4`R`I<MU!9y{KUNkMRwTknnC9W1rRGfo|t8qwPr%Q{"
    "x5)Q9PtoNdm-cjvhx<T};py$HS*+5!$*78@<i(SlbG@4(3UjQu_yEW~~2_O3Hel;xW@BO=0q^4_^XTUyApSz{ACX"
    "f~sH)v#T_8RX=&<B8fgYSZ4g4cmdfxmz=AWtJVA|D|K!0*D>!N1^6;~qhGp)c4Yycf*HS;g1(lWiD<DCL-OEs3I?"
    "nLIyhqh#2g0w?!l_QAAF)yT8el|x#0HAL2~h#(GsNv2(jCj0nM&FpDBhvq1pDwb>hoBzGO{*gEvPHP{V49YocvnO"
    "&Bujb$5P^f&>ga^(3<NXnO67&J!-"
    "r&37nBaBbQs6J(49L^SjmSsH0r0!<b?`5^)3`^_UFZw;2=4`Rsa5IA>))76+F9n0{u7a63Y-"
    "UbuATkskJqT!W(7r(O0&j+^npZr{-"
    "N}EW7&H4>U+!Sw{ybD@YK$(FBe1Ux07SMlX)T~zbWjPEV!2)dTAM_D)4unquxC)<T>NL8u}yjB<KUcy}@_EF~RG="
    "rNCdn8IY%u8<CHY1K@Yz>)>B-r*V&<yU-Wx5#9^t^5Espl-"
    "?swbZqNyhx)HZOe%W`&x~0*w9D?^1sf?vD(9KB;AH(S@<<bY-o?b((@c$dd1Z>^vijK0dz0meGh4^lpCd!-"
    "@`o0jx;L3_Z!}kOUcQ#lH=<WVe}tX{eE_&O_%1jmcpbPD_{%tF_<x>8ZbUvp4uIc<uY-TVoyI+a?m}O%M|dxo%d>"
    "&XfE|`Wq@Ur?%<bz>R>z9WV%LX|o!<L7){k}BM$rp9jy(+|rSCawq%V3?e5BrDN4a1+{qna)RiYmIxOh<Q%!@#ho"
    "%OVZN7Q?~heh9rUJd;bdJ^;j;NIZ7;F#cb;8Nf(;0(yq<GJzw`3N}xeiyzD{sng$_XxTReZd~#y<jd4X_60k%R_0"
    "TcG_TyVK_OwemMH3MV`_7#|=v*LP&giTHHR@K+08rrJ5QOM#VY;m)nYU*qfzAH*a}!Xz`0gm!Bv7#`|gXu;?4ntD"
    "!$aPl7%G+#7rs922|_TnhXJoB??nxe@sYIRJhaz7GBccN+Hyx(j{59^t)UE?=}olv^8)(1!Bv%dLf)%-"
    "ZT~foNa^g^PaRdF2vHs>!;$YeEi__riziQ--wI60?)TLhB>QVZuANJ2fHWy*>JA^swj~(W{|9LQjG|0NfjV7aSA3"
    "4qOWS1)KqS8o3eq2sr?L7rqYu1$P?v2)YY>!5-"
    "ngU@nJO`^^j3XhU9xwoxlAtQe>4^ZMGmb6EOYBk?a$TbS47bw4XA71=(n#M1VkJDBUXm2Fv#TI2sSptnaqjUE<#B"
    "YHLTN9ak=2Y`En?}B53*MUoczko9!Pa`)XA0Y?8?~ePr|N9r*Y1|{|F7yR^g!h8E?8uyFnA0ChYv!JhmfYY^3a_m"
    ";UrP?7vDCs(j>omw%lhS<Iv)-Vmo8pm@8UoHp9lVDKyQzJ8a*ufM)YdvkI<8#4*>TD-"
    "v!46uLG9?e*tGeo<?p&K0*$F--WM(f5Dx`J%a8+U$94bFPO{HPpjv-"
    "3;B`Ri18cGL=L@`)vjN^)`)F3o3bllNf0fRiHq4j!E5~g7yRde{~6HRqn}0(i@p)P8u}yjB<KUcy}@_EF~RG=rNC"
    "dn8IY%u8<CHY1K@Yz>)>B-r*V&<yU-Wx5#9^tGIv6&hf3BF5;r)Ud$Gfqbw6oeT-_Bx($6;Z-3SXA|1Svtzu-"
    "R){Lg^i9{n_WSoDqP)zBZICqW+o?hU>RjtO1|E(QJq&VW3P+=zUH900!yUkCq!JB@n;-"
    "G#njkMLeFm)9jio6DcuQB3B1?;TS1EaZ3p>12&f<Nr(Hzaaeog8w}5KLdJu^wa2J(Kn)3Lw|&x1bqOwH~20%CU_m"
    "V6!;4`1M)O-Bk~b)0Q@d|9sCRKH0}{}7y5!d!h6A77TUF}>RTL6Z-"
    "2GPUla2i|Bnp+OX0sD{QrXgJn%mQdVBQK=wZ<}qE|zIgq{R_0Jt~!E;uH59k>+u3pfMvG;$;I5pn?hE_@yQ3+^=T"
    "5p);&f<3}}!Cdqo>Zo+ec+>v?L(Ay$"
)

# Uncompressed SHA-256: 916b84b935262e554f965f9022230bf9d3043f6d69cc8a851abf8d39c563e6fd
_FADE_FOGFREE_DATA = (
    "c-lRa>rWGQ7==M1qDU|_kcfb^s9P14l~EvSPb0Mi6w3-"
    "?v=ry)R%nr+Lyc}>;3ZM1go}Y7myv0JDSm;t)`9$}C{8KJK(MgUrV%k#C)h@1qm_Mn`3Fu;p7Wfhl5I8q4l|UcJ5"
    "5n_-53jB_tH7ljv0?-wpG6`7biXMlr(k?Lz!c3c+&62TYl6*o!Wu<ZNg_sC9}}{asHV6gc}=|wq-qbJb?UPII|D_"
    "jaJK?r*HlWv5vznbiOG-;pK!p?|lj+e@f|(Zpp;jq^hxoSq)a%%8E}4h3xj|PY?88s+fl{A~kDm0ZZs}%#1G-"
    "uvewlE&C^SLH^HV*`-P#P$G@~{s@=Z2U4@Et=Wund5!&;ewv5)3WM=|`Fl~2^A5LJA;-"
    "NkyFr>2%iNh#b1yH0>CbXx17;cP`aFNkZ)+?Qg!c{R*Ka~@(aGGWTQ?!?h1&DOpi2~N3Feok@fr1{+(*?e(Vh+8A"
    "tCTg#(aGmei=^Q(^C-"
    "+foES(OJy?q&2dL}+Vd;6W^+WjDOSlWYYHSfLn!_d&27l%_%c<X|BGHr2qFSL6Xnh%GU7meDVG<VT$|4A!r9=8Kj"
    "Q+gK>MnptIqctR3hH%<B3fe@nP#-!dfK0Nf*7d7Gdww(aJ;jPhooK6rA1bSzTtYU!|;hkuPzezLa}du5Jl?r-"
    "NF+tI->ZF?t~`XZ3zH*3G^Y8Xw6pS+ZO9!6yvI8k-~AGt}(vwD`aR+ss-FO)F<U=@xe*zQlq0Qm$smw4&PdGlFjN"
    "Vx#_R#At&;fWlgA^#4ISH}VZ8r?vKUTPn0EzurE=OF_p}Tbw!ID#!=gjrbA=>Pxw8;=#13DkB6#K0-"
    ";h9@|?>_g*n8ku>&WjCV~Qv$n-"
    "+R52HEtHV?;xzPjijC`Qoh%a%VzLY!S)Bdg0V#Gwqy@_gv95=h8IebL~{9=|D+F3EWud8_6$R3ct<Qe%uyAfaFKz"
    "%9qoO?hs9(e$-^)9>k?ryYq&Mpzy!q~_1k6y!7M;CuV{*q_p1MNn9i39beT(5a4CqQ`=tk-h>p-"
    "szFF22XW+pib@M!z6`$ushSb|b#Tf%;M|liU)15n6=fUOW0aTs5E{)8FV9<S%(fKG1H&mpD*g%6-<Ua9bWe!T-"
    "kXUU~"
)

# Uncompressed SHA-256: 1d421c8d7627297c1a1c3223014597c8620615eb2d1ddd419ae757d6b89839d0
_FADE_FOGGY_DATA = (
    "c-"
    "lRa{ZGsR9LF`<vh_gaDM^tNO)kEjh<lgo!PDh3(VU0XvC0*4vA9KtM7m0J$9afyoUVpsE!@7Suu(*p5;d1+Q*AkO"
    "Qu*og{RiHA@AZDY_4ik>@!DG0%ttQ$t<FZMbEn|KU>5wQ)dkGk(+#aZpC4GIfNM?MxMeO+VJR6Qbbj#w`%A~m4oa"
    "H9msYg%n4i!OwN+z|1^acuE@jDj3@9}{jH4CBcL$7@Qq#c8>~P4>j)ci~t=7>fgiN++gLj)6?v-z%o-281-"
    "MglpJ(P}cRo?+cQWA1rW_@+Pk%|pRS1_X<<wH=;zV%LtWutyKGY*DX@ZGFRc^IkUa>rj(G?}{nU`rKs$wh96Gv{Q"
    ")25m$770-LK>g?ds*4@FI5`#u@ul=)-"
    "F!UvcJ0+BhaMNSmqPdL;;0L8;_SEv(3>~(7h_*wJQ^<$r`LSGI$}RqKEjjw14+6zuEnkAAc${3{a{8iG7}qSiJZ|"
    "1nbTKn@&Oa8QA-KVQ(%ujpx#g$Q=dOqSx#(kO+IVci`AU^d`bU`eO>(wgC*e3yU&>unw9g}<yc+h~-"
    "+g4R$ABpuv?coKv2smi?m$TaCWl9C?^m2emA9SNO??~Fmm~zbygtO9h#JhR9lILs?(>DFm6tib#DV%!Zqd-"
    "e>7O?B3$qPQjshPMs;3QSmB?q~S)sRbeV;$lj1%tJW{Sa6_l#(~<I4)hY`$Bi{47i@9pdeD7jU}~U*bT0Dff^3Rh"
    "M3K6|D<1&KN3lVR;fDI~sBvlSU@W&6YBVd!zxMjk{Ux+f@zUe+NNQ<KVG8FOTzqb|b#Tf%;OedT~d4%oZJX!Mys#"
    "t|C-acy{iPo<e+C>DYygl>ITg$+s{w62DrW)F-RUIM2uj+Ku=U2kJ|?EuBWI#_bARCmxQMMriQL{L<<p-"
    "^Pv#AI%J(+K6THM*le3P0nBPjC`Qoh%a%VzLe`!82$CZdlP0X*?NNUQn9^VtyA_f>(IQ_v@=I(;C?~=l4s-"
    "t?M8fw1NEicBty+qrI!N7-f5b`Zbz_|f#TC^AFH^((J#nf@{D|--"
    "H0!7puUuwy`#J{eYpg6!_J{zwFTUd>2LH4@|Qd#A80q?OB|>#<yyt{{rSCS{0DcyLG="
)


def _decode_arrays(encoded: str, shapes: tuple[tuple[int, ...], ...]):
    raw = zlib.decompress(base64.b85decode(encoded.encode("ascii")))
    values = np.frombuffer(raw, dtype="<f8")
    arrays = []
    offset = 0
    for shape in shapes:
        size = int(np.prod(shape))
        array = values[offset : offset + size].reshape(shape).copy()
        array.setflags(write=False)
        arrays.append(array)
        offset += size
    if offset != values.size:
        raise RuntimeError("Invalid bundled natural-scene model data")
    return tuple(arrays)


NIQE_PRISTINE_MEAN, NIQE_PRISTINE_COVARIANCE = _decode_arrays(
    _NIQE_DATA, ((36,), (36, 36))
)
FADE_FOGFREE_MEAN, FADE_FOGFREE_COVARIANCE = _decode_arrays(
    _FADE_FOGFREE_DATA, ((12,), (12, 12))
)
FADE_FOGGY_MEAN, FADE_FOGGY_COVARIANCE = _decode_arrays(
    _FADE_FOGGY_DATA, ((12,), (12, 12))
)

del _NIQE_DATA, _FADE_FOGFREE_DATA, _FADE_FOGGY_DATA
