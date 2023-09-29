# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    ft_progress.py                                    :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <contact@charles-mariot.fr>       +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/09/27 11:13:10 by cmariot          #+#    #+#              #
#    Updated: 2023/09/28 15:28:58 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #

from time import time
from os import get_terminal_size


def ft_progress(iterable,
                length=get_terminal_size().columns - 4,
                fill='█',
                empty='░',
                print_end='\r'):
    """
    Progress bar generator.
    """

    def get_elapsed_time_str(elapsed_time):
        """
        Return the elapsed time as str.
        """
        if elapsed_time < 60:
            return f'[Elapsed-time {elapsed_time:.2f} s]'
        elif elapsed_time < 3600:
            return f'[Elapsed-time {elapsed_time / 60:.0f} m]'
        else:
            return f'[Elapsed-time {elapsed_time / 3600:.0f} h]'

    def get_eta_str(eta):
        """
        Return the Estimed Time Arrival as str.
        """
        if eta == 0.0:
            return ' [DONE]                         '
        elif eta < 60:
            return f' [{eta:.0f} s remaining]       '
        elif eta < 3600:
            return f' [{eta / 60:.0f} m remaining]  '
        else:
            return f' [{eta / 3600:.0f} h remaining]'

    try:
        print()
        total = len(iterable)
        start = time()
        for i, item in enumerate(iterable, start=1):
            elapsed_time = time() - start
            et_str = get_elapsed_time_str(elapsed_time)
            eta_str = get_eta_str(elapsed_time * (total / i - 1))
            filled_length = int(length * i / total)
            percent_str = f'[{(i / total) * 100:6.2f} %] '
            progress_str = str(fill * filled_length
                               + empty * (length - filled_length))
            counter_str = f'  [{i:>{len(str(total))}}/{total}] '
            bar = ("\033[F\033[K  " + progress_str + "\n"
                   + counter_str
                   + percent_str
                   + et_str
                   + eta_str)
            print(bar, end=print_end)
            yield item
        print()
    except Exception:
        print("Error: ft_progress")
        return None
