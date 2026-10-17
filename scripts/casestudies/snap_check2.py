import json,pickle,sys,numpy as np
from scipy.spatial import cKDTree
S=sys.argv[1]
R=json.load(open(S+'/amazon_raw/route_data.json'))
def proj(ll,lat0):
    return np.c_[(ll[:,1])*111320*np.cos(np.radians(lat0)), ll[:,0]*110540]
def sample_elev(lat,lng):
    out=np.full(len(lat),np.nan)
    tiles={}
    for i,(a,b) in enumerate(zip(lat,lng)):
        la=int(np.floor(a)); lo=int(np.floor(b))
        k=(la,lo)
        if k not in tiles:
            n='N%02dW%03d'%(la,-lo)
            tiles[k]=np.fromfile('%s/srtm/%s.hgt'%(S,n),dtype='>i2').reshape(3601,3601)
        t=tiles[k]
        r=(la+1-a)*3600; c=(b-lo)*3600
        r0=int(r); c0=int(c); fr=r-r0; fc=c-c0
        v=(t[r0,c0]*(1-fr)*(1-fc)+t[r0+1,c0]*fr*(1-fc)+t[r0,c0+1]*(1-fr)*fc+t[r0+1,c0+1]*fr*fc)
        out[i]=v
    return out
for name,pre,pk in (('Seattle','DSE','seattle_graph.pkl'),('Chicago','DCH','chicago_graph.pkl')):
    G=pickle.load(open(S+'/'+pk,'rb'))
    lat0=G['latlon'][:,0].mean()
    tree=cKDTree(proj(G['latlon'],lat0))
    pts=[(v['lat'],v['lng']) for r in R.values() if r['station_code'].startswith(pre) for v in r['stops'].values()]
    rng=np.random.default_rng(0); pts=np.array(pts)[rng.choice(len(pts),20000,replace=False)]
    d,_=tree.query(proj(pts,lat0))
    print(name,'stops->nearest road node dist m pct 50/75/90/99:',np.percentile(d,[50,75,90,99]).round(1),'frac>100m',(d>100).mean().round(4))
    e=sample_elev(G['latlon'][::50,0],G['latlon'][::50,1])
    print(name,'road node elev m pct 0/25/50/75/100',np.nanpercentile(e,[0,25,50,75,100]).round(1),'nan',np.isnan(e).sum(), 'void(-32768)',(e<-1000).sum())
