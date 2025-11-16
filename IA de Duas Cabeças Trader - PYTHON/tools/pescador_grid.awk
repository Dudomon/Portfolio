BEGIN{
  FS=",";
  close_idx=-1;high_idx=-1;low_idx=-1;row=0;
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
  } else { cross_up[row]=0; cross_dn[row]=0; }
}
END{
  n=row;
  # Collect event indices
  ucnt=0; dcnt=0;
  for(i=2;i<=n;i++){
    if(cross_up[i]){ up_idx[++ucnt]=i; }
    if(cross_dn[i]){ dn_idx[++dcnt]=i; }
  }
  # Parse SL/TP lists
  split(SLV, sls, ","); slc=0; for(i in sls){ val=sls[i]+0.0; if(val>0){ sl[++slc]=val; }}
  split(TPV, tps, ","); tpc=0; for(i in tps){ val=tps[i]+0.0; if(val>0){ tp[++tpc]=val; }}
  if(H<=0) H=10;
  printf("Events: up=%d, dn=%d | horizon=%d\n", ucnt, dcnt, H);
  best_ev=-1e9; best_sl=0; best_tp=0; best_hit=0; best_trials=0;
  for(si=1; si<=slc; si++){
    for(ti=1; ti<=tpc; ti++){
      s=sl[si]; t=tp[ti];
      total=0; wins=0;
      # Long side
      for(k=1;k<=ucnt;k++){
        i=up_idx[k]; entry=closev[i]; end=(i+H<=n)?i+H:n;
        hit_tp=0; hit_sl=0;
        for(j=i+1;j<=end;j++){
          if(highv[j] >= entry + t){ hit_tp=1; break; }
          if(lowv[j]  <= entry - s){ hit_sl=1; break; }
        }
        wins += (hit_tp==1 && hit_sl==0)?1:0;
        total++;
      }
      # Short side
      for(k=1;k<=dcnt;k++){
        i=dn_idx[k]; entry=closev[i]; end=(i+H<=n)?i+H:n;
        hit_tp=0; hit_sl=0;
        for(j=i+1;j<=end;j++){
          if(lowv[j] <= entry - t){ hit_tp=1; break; }
          if(highv[j] >= entry + s){ hit_sl=1; break; }
        }
        wins += (hit_tp==1 && hit_sl==0)?1:0;
        total++;
      }
      hit = (total>0)? wins/total : 0;
      ev = t*hit - s*(1-hit);
      printf("SL=%.2f TP=%.2f | hit=%5.1f%% | EV~%.3f | trials=%d\n", s, t, hit*100.0, ev, total);
      if(ev>best_ev){ best_ev=ev; best_sl=s; best_tp=t; best_hit=hit; best_trials=total; }
    }
  }
  printf("\nBest by EV: SL=%.2f TP=%.2f | hit=%5.1f%% | EV~%.3f | trials=%d\n", best_sl, best_tp, best_hit*100.0, best_ev, best_trials);
}
