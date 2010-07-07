//    Copyright 2010 Chris Jang
//
//    This file is part of GATLAS.
//
//    GATLAS is free software: you can redistribute it and/or modify
//    it under the terms of the GNU Lesser General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    GATLAS is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU Lesser General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public License
//    along with GATLAS.  If not, see <http://www.gnu.org/licenses/>.

#include <iostream>
#include <string>
#include <unistd.h>
#include "GatlasBenchmark.hpp"

#include "using_namespace"

using namespace std;

bool parseOpts(int argc, char *argv[], string& journalFile, bool& deleteTimes) {
    int opt;
    while ((opt = getopt(argc, argv, "hdj:")) != -1) {
        switch (opt) {
            case ('h') :
                cerr << "usage: " << argv[0]
                     << " -j journalFile"
                        " [-d] [-h]" << endl
                     << "\t-j journal file" << endl
                     << "\t-d delete benchmark time records for good kernels (default is to keep them)" << endl
                     << "\t-h help" << endl;
                exit(1);
            case ('j') : journalFile = optarg; break;
            case ('d') : deleteTimes = true; break;
        }
    }
    // minimal validation of options
    bool rc = true;
    if (journalFile.empty()) {
        cerr << "error: journal file must be specified" << endl;
        rc = false;
    }
    return rc;
}

int main(int argc, char *argv[])
{
    string journalFile;
    bool deleteTimes = false;

    if (!parseOpts(argc, argv, journalFile, deleteTimes))
        exit(1);

    Journal journal(journalFile);

    if (journal.loadMemo()) {
        const size_t numGoodKernels = journal.memoGood();
        const int numBadKernels = journal.purgeMemo(deleteTimes);
        cout << "journal file " << journalFile
             << " has " << numBadKernels << " bad keys, "
             << (deleteTimes ? "deleted " : "preserved ")
             << numGoodKernels << " good keys"
             << endl;
    } else {
        cerr << "error: could not load journal file " << journalFile << endl;
    }

    return 0;
}
