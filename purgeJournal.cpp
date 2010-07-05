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

bool parseOpts(int argc, char *argv[], string& journalFile) {
    int opt;
    while ((opt = getopt(argc, argv, "hj:")) != -1) {
        switch (opt) {
            case ('h') :
                cerr << "usage: " << argv[0]
                     << " -j journalFile"
                        " [-h]" << endl
                     << "\t-j journal file" << endl
                     << "\t-h help" << endl;
                exit(1);
            case ('j') : journalFile = optarg; break;
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

    if (!parseOpts(argc, argv, journalFile))
        exit(1);

    Journal journal(journalFile);

    if (journal.loadMemo()) {
        const size_t numGoodKernels = journal.memoGood();
        const int numBadKernels = journal.purgeMemo();
        cout << "purged journal file " << journalFile
             << " of " << numGoodKernels
             << " good keys, leaving " << numBadKernels
             << " bad keys"
             << endl;
    } else {
        cerr << "error: could not load journal file " << journalFile << endl;
    }

    return 0;
}
