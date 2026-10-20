"""Execute actual report cells through IPython without a network kernel process."""
import contextlib
import io
import os
from pathlib import Path

import nbformat as nb
from IPython.core.interactiveshell import InteractiveShell
from traitlets.config import Config


def execute_in_process(book,directory):
    cache=Path(directory)/'.cache'
    for variable,name in [('MPLCONFIGDIR','matplotlib'),('IPYTHONDIR','ipython')]:
        target=cache/name;target.mkdir(parents=True,exist_ok=True)
        os.environ[variable]=str(target)
    import matplotlib
    matplotlib.use('Agg')
    config=Config();config.HistoryManager.hist_file=':memory:'
    shell=InteractiveShell.instance(config=config)
    previous_publisher=shell.display_pub.publish
    original_directory=os.getcwd()
    outputs=[]

    def publish(data,metadata=None,**kwargs):
        outputs.append(nb.v4.new_output('display_data',data=data,metadata=metadata or {}))

    shell.display_pub.publish=publish
    try:
        os.chdir(directory)
        count=0
        for cell in book.cells:
            if cell.cell_type!='code':
                continue
            count+=1
            outputs=[]
            stdout,stderr=io.StringIO(),io.StringIO()
            with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
                result=shell.run_cell(cell.source,store_history=False)
            for name,stream in [('stdout',stdout),('stderr',stderr)]:
                value=stream.getvalue()
                if value:
                    outputs.append(nb.v4.new_output('stream',name=name,text=value))
            cell.execution_count=count
            cell.outputs=outputs
            error=result.error_before_exec or result.error_in_exec
            if error is not None:
                raise RuntimeError(f'Report cell {count} failed: {error}') from error
    finally:
        shell.display_pub.publish=previous_publisher
        os.chdir(original_directory)
