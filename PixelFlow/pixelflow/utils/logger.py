import os
import logging


class PathSimplifierFormatter(logging.Formatter):
    def format(self, record):
        record.short_path = os.path.relpath(record.pathname)
        return super().format(record)


def setup_logger(log_directory, experiment_name, process_rank, source_module=__name__):
    handlers = [logging.StreamHandler()]

    if process_rank == 0:
        log_file_path = os.path.join(log_directory, f"{experiment_name}.log")
        handlers.append(logging.FileHandler(log_file_path))

    log_formatter = PathSimplifierFormatter(
        fmt='[%(asctime)s %(short_path)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    for handler in handlers:
        handler.setFormatter(log_formatter)

    logging.basicConfig(level=logging.INFO, handlers=handlers)
    return logging.getLogger(source_module)
