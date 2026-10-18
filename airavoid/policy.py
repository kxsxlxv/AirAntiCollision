"""Temporal confirmation for warnings and, more conservatively, apparent clearance."""


class TemporalPolicy:
    def __init__(self,warning_seconds=.24,nonthreat_seconds=1.2):
        self.warning_seconds=warning_seconds
        self.nonthreat_seconds=nonthreat_seconds
        self.pending={}

    def apply(self,track_id,t,estimate):
        e=estimate.copy();state=e['state'];reason=e['reason']
        e['candidate_state']=state;e['candidate_reason']=reason
        if state=='uncertain':
            self.pending.pop(track_id,None)
            return e
        key=(state,reason)
        old=self.pending.get(track_id)
        if old is None or old[0]!=key or t-old[2]>.3:
            old=(key,t,t)
        self.pending[track_id]=(key,old[1],t)
        elapsed=t-old[1]
        needed=self.warning_seconds if state=='threat' else self.nonthreat_seconds
        e['confirmation_elapsed_s']=elapsed;e['confirmation_required_s']=needed
        if elapsed+1e-6<needed:
            e['state']='uncertain';e['reason']='confirm_warning' if state=='threat' else 'confirm_non_threat'
        return e
