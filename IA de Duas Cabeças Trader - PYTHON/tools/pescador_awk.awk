BEGIN{
  close_idx=-1;high_idx=-1;low_idx=-1;row=0;
  split(HLIST,Hs," ");
  hc=0;
  for(i in Hs){hv=Hs[i]; horiz_list[++hc]=hv; horiz[hv]=hv; up_mfe[hv]=0; up_mae[hv]=0; up_cnt[hv]=0; dn_mfe[hv]=0; dn_mae[hv]=0; dn_cnt[hv]=0;}
}
NR==1{
  for(i=1;i<=NF;i++){
    if($i=="close") close_idx=i;
    else if($i=="high") high_idx=i;
    else if($i=="low") low_idx=i;
  }
  if(close_idx<0||high_idx<0||low_idx<0){ print "Missing columns" > "/dev/stderr"; exit 1 }
  next;
}
{
  row++;
  closev[row]=$(close_idx)+0.0;
  highv[row]=$(high_idx)+0.0;
  lowv[row]=$(low_idx)+0.0;
  if(row==1){ ema[row]=closev[row]; } else { ema[row]=ALPHA*closev[row]+(1-ALPHA)*ema[row-1]; }
  if(row>1){
    slope[row]=ema[row]-ema[row-1];
    cross_up[row]=(closev[row-1]<ema[row-1] && closev[row]>=ema[row] && slope[row]>0)?1:0;
    cross_dn[row]=(closev[row-1]>ema[row-1] && closev[row]<=ema[row] && slope[row]<0)?1:0;
  } else { slope[row]=0; cross_up[row]=0; cross_dn[row]=0; }
}
END{
  n=row;
  total_up=0; total_dn=0;
  for(i=2;i<=n;i++){
    if(cross_up[i]){ ev_idx_up[++total_up]=i; }
    if(cross_dn[i]){ ev_idx_dn[++total_dn]=i; }
  }
  printf("Events: cross_up=%d, cross_dn=%d\n", total_up, total_dn);
  for(hi=1; hi<=hc; hi++){
    hh=horiz_list[hi];
    # long events
    for(k=1;k<=total_up;k++){
      i=ev_idx_up[k]; entry=closev[i]; end=(i+hh<=n)?i+hh:n;
      if(end<=i) continue;
      maxh=highv[i+1]; minh=lowv[i+1];
      for(j=i+1;j<=end;j++){ if(highv[j]>maxh) maxh=highv[j]; if(lowv[j]<minh) minh=lowv[j]; }
      mfe=(maxh-entry); if(mfe<0) mfe=0;
      mae=(entry-minh); if(mae<0) mae=0;
      up_mfe[hh]+=mfe; up_mae[hh]+=mae; up_cnt[hh]++;
    }
    # short events
    for(k=1;k<=total_dn;k++){
      i=ev_idx_dn[k]; entry=closev[i]; end=(i+hh<=n)?i+hh:n;
      if(end<=i) continue;
      maxh=highv[i+1]; minh=lowv[i+1];
      for(j=i+1;j<=end;j++){ if(highv[j]>maxh) maxh=highv[j]; if(lowv[j]<minh) minh=lowv[j]; }
      mfe=(entry-minh); if(mfe<0) mfe=0;
      mae=(maxh-entry); if(mae<0) mae=0;
      dn_mfe[hh]+=mfe; dn_mae[hh]+=mae; dn_cnt[hh]++;
    }
    um= (up_cnt[hh]>0)? up_mfe[hh]/up_cnt[hh] : 0;
    ua= (up_cnt[hh]>0)? up_mae[hh]/up_cnt[hh] : 0;
    dm= (dn_cnt[hh]>0)? dn_mfe[hh]/dn_cnt[hh] : 0;
    da= (dn_cnt[hh]>0)? dn_mae[hh]/dn_cnt[hh] : 0;
    printf("H=%d | LONG MFE=%.4f MAE=%.4f (n=%d) | SHORT MFE=%.4f MAE=%.4f (n=%d)\n", hh, um, ua, up_cnt[hh], dm, da, dn_cnt[hh]);
  }
}
