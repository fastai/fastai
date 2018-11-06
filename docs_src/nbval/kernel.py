"""
pytest ipython plugin modification

Authors: D. Cortes, O. Laslett, T. Kluyver, H. Fangohr, V.T. Fauske

"""

import os
import logging
from pprint import pformat

try:
    from Queue import Empty
except:
    from queue import Empty

# Kernel for jupyter notebooks
from jupyter_client.manager import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import ipykernel.kernelspec


CURRENT_ENV_KERNEL_NAME = ':nbval-parent-env'

logger = logging.getLogger('nbval')
# Uncomment to debug kernel communication:
# logger.setLevel('DEBUG')
# logging.basicConfig(format="[%(asctime)s - %(name)s - %(levelname)s] %(message)s")


class NbvalKernelspecManager(KernelSpecManager):
    """Kernel manager that also allows for python kernel in parent environment
    """

    def get_kernel_spec(self, kernel_name):
        """Returns a :class:`KernelSpec` instance for the given kernel_name.

        Raises :exc:`NoSuchKernel` if the given kernel name is not found.
        """
        if kernel_name == CURRENT_ENV_KERNEL_NAME:
            return self.kernel_spec_class(
                resource_dir=ipykernel.kernelspec.RESOURCES,
                **ipykernel.kernelspec.get_kernel_dict())
        else:
            return super(NbvalKernelspecManager, self).get_kernel_spec(kernel_name)


def start_new_kernel(startup_timeout=60, kernel_name='python', **kwargs):
    """Start a new kernel, and return its Manager and Client"""
    logger.debug('Starting new kernel: "%s"' % kernel_name)
    km = KernelManager(kernel_name=kernel_name,
                       kernel_spec_manager=NbvalKernelspecManager())
    km.start_kernel(**kwargs)
    kc = km.client()
    kc.start_channels()
    try:
        kc.wait_for_ready(timeout=startup_timeout)
    except RuntimeError:
        logger.exception('Failure starting kernel "%s"', kernel_name)
        kc.stop_channels()
        km.shutdown_kernel()
        raise

    return km, kc


class RunningKernel(object):
    """
    Running a Kernel a Jupyter, info can be found at:
    http://jupyter-client.readthedocs.org/en/latest/messaging.html

    The purpose of this class is to encapsulate interaction with the
    jupyter kernel. Thus any changes on the jupyter side to how
    kernels are started/managed should not require any changes outside
    this class.

    """
    def __init__(self, kernel_name, cwd=None):
        """
        Initialise a new kernel
        specify that matplotlib is inline and connect the stderr.
        Stores the active kernel process and its manager.
        """

        self.km, self.kc = start_new_kernel(
            kernel_name=kernel_name,
            stderr=open(os.devnull, 'w'),
            cwd=cwd,
        )

        self._ensure_iopub_up()

    def _ensure_iopub_up(self):
        total_timeout = 30
        individual_timeout = 1
        shell_timeout = 10
        for _ in range(total_timeout // individual_timeout):
            msg_id = self.kc.kernel_info()

            try:
                self.await_reply(msg_id, timeout=shell_timeout)
            except Empty:
                raise RuntimeError('Kernel info reqest timed out after %d seconds!' % shell_timeout)

            try:
                self.await_idle(msg_id, individual_timeout)
            except Empty:
                continue
            else:
                # got IOPub
                break
        else:
            raise RuntimeError("Wasn't able to establish IOPub after %d seconds." % total_timeout)

    def get_message(self, stream, timeout=None):
        """
        Function is used to get a message from the iopub channel.
        Timeout is None by default
        When timeout is reached
        """
        try:
            if stream == 'iopub':
                msg = self.kc.get_iopub_msg(timeout=timeout)
            elif stream == 'shell':
                msg = self.kc.get_shell_msg(timeout=timeout)
            else:
                raise ValueError('Invalid stream specified: "%s"' % stream)
        except Empty:
            logger.debug('Kernel: Timeout waiting for message on %s', stream)
            raise
        logger.debug("Kernel message (%s):\n%s", stream, pformat(msg))
        return msg

    def execute_cell_input(self, cell_input, allow_stdin=None):
        """
        Executes a string of python code in cell input.
        We do not allow the kernel to make requests to the stdin
             this is the norm for notebooks

        Function returns a unique message id of the reply from
        the kernel.
        """
        if cell_input:
            logger.debug('Executing cell: "%s"...', cell_input.splitlines()[0][:40])
        else:
            logger.debug('Executing empty cell')
        return self.kc.execute(cell_input, allow_stdin=allow_stdin, stop_on_error=False)

    def await_reply(self, msg_id, timeout=None):
        """
        Continuously poll the kernel 'shell' stream for messages until:
         - It receives an 'execute_reply' status for the given message id
         - The timeout is reached awaiting a message, in which case
           a `Queue.Empty` exception will be raised.
        """
        while True:
            msg = self.get_message(stream='shell', timeout=timeout)

            # Is this the message we are waiting for?
            if msg['parent_header'].get('msg_id') == msg_id:
                if msg['content']['status'] == 'aborted':
                    # This should not occur!
                    raise RuntimeError('Kernel aborted execution request')
                return

    def await_idle(self, parent_id, timeout):
        """Poll the iopub stream until an idle message is received for the given parent ID"""
        while True:
            # Get a message from the kernel iopub channel
            msg = self.get_message(timeout=timeout, stream='iopub') # raises Empty on timeout!

            if msg['parent_header'].get('msg_id') != parent_id:
                continue
            if msg['msg_type'] == 'status':
                if msg['content']['execution_state'] == 'idle':
                    break

    def is_alive(self):
        if hasattr(self, 'km'):
            return self.km.is_alive()
        return False

    # These options are in case we wanted to restart the nb every time
    # it is executed a certain task
    def restart(self):
        """
        Instructs the kernel manager to restart the kernel process now.
        """
        logger.debug('Restarting kernel')
        self.km.restart_kernel(now=True)

    def interrupt(self):
        """
        Instructs the kernel to stop whatever it is doing, and await
        further commands.
        """
        logger.debug('Interrupting kernel')
        self.km.interrupt_kernel()

    def stop(self):
        """
        Instructs the kernel process to stop channels
        and the kernel manager to then shutdown the process.
        """
        logger.debug('Stopping kernel')
        self.kc.stop_channels()
        self.km.shutdown_kernel(now=True)
        del self.km

    @property
    def language(self):
        if self.km.kernel_spec is None:
            return None
        return self.km.kernel_spec.language
