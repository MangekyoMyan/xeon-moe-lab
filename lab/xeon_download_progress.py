"""Optional status records for the local resumable downloaders; no network calls."""
import json
import os
from pathlib import Path
import tempfile
import time


def write_progress(directory, repository, revision, phase, item=None, done=0):
    temporary=None
    try:
        directory=Path(directory)
        if not directory.is_dir():return False
        pid=os.getpid()
        start=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
        record={'schema':1,'repository':repository,'revision':revision,'phase':phase,
                'unix':time.time(),'pid':pid,'start_ticks':start,
                'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'method':'sequential-curl-part','file':item.get('name') if item else None,
                'done_bytes':done,'total_bytes':item.get('size') if item else None}
        with tempfile.NamedTemporaryFile(mode='w',dir=directory,prefix='.download-progress-',suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name)
            json.dump(record,stream);stream.write('\n')
        temporary.replace(directory/'download-progress.json')
        return True
    except (OSError,ValueError,IndexError):
        return False
    finally:
        if temporary is not None:
            try:temporary.unlink(missing_ok=True)
            except OSError:pass
