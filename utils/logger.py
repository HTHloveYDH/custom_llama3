import sys
import re

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):      
        if self.should_log(message):
            if message != '\n':
                self.log.write(message+'\n')
                self.log.flush()
        else:
            # re: remove \x00-\x1F\x7F code
            a = str(message).split()[1]
            clean_a = re.sub(r"[\x00-\x1F\x7F]", "", str(a))
            # example: clean_a: 22/1000
            nums = clean_a.split('/')
            if len(nums) > 1 and nums[0] == nums[1]:
                self.log.write(message+'\n')
                self.log.flush()
        self.terminal.write(message)

        

    def should_log(self, message):
        keywords2filter = ['- ETA:', '- loss:']
        return not any(keyword in message for keyword in keywords2filter)

    def flush(self):
        pass



# sys.stdout = Logger('a1.log', sys.stdout)
# sys.stderr = Logger('a.log_file', sys.stderr)
