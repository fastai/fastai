import sys
import re




def expand(filename):

    f = open(filename, "r")
    contents = f.read()

    regex_inside = r"\{\{(.*?)\}\}"
    regex_outside = r"(^|\}\})(.*?)(\{\{|$)"

    within = re.finditer(regex_inside, contents, re.MULTILINE | re.DOTALL)
    outside = re.finditer(regex_outside, contents, re.MULTILINE | re.DOTALL) 

    for matchNum, match in enumerate(within):
        for groupNum in range(0, len(match.groups())):
            group = match.group(1)
            if group.startswith("class"):
                classname = re.search(r" (.*?),", group).groups()[0]
                params = re.search(r",(.*)", group).groups()[0]
                print('<h2 id="' + classname + '" class="class">Class: ' + classname + '(<span class="params">' + params + '</span></h2>')

            print (match.group(1))

#    split = re.split(regex_inside, contents)
#
#    for i, item in enumerate(split):



if __name__ == '__main__':

    expand(sys.argv[1])

