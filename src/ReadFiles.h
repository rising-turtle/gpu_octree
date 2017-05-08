#ifndef READFILES_H
#define READFILES_H

#include <string>
#include <map>

using namespace std;

// each record corresponds to depth timestamp 

class CReadFiles
{
public:
    typedef double Key; 
    typedef string Value;
    typedef map<Key, Value> Record;
    typedef Record::iterator r_iter; 

public:
    CReadFiles();
    CReadFiles(string dir);
    ~CReadFiles();
    void test();
    void initDir(string dir);
    bool readRecord(string& vis_f, string& dth_f, string& gt_v);
    bool getCurGtStr(double & timestamp , string& gt_v);
    double getCurQueryTime() {return m_cur_stamp;}
private:
    void clearRecords();
    bool readFiles(string, Record&);
    void dumpFiles(string, Record&);
    bool getCloseValue(r_iter&, r_iter, Key, Value&);
    
    map<double, string> m_gtRecord;
    map<double, string> m_visRecord;
    map<double, string> m_dptRecord;
    
    string m_dir;
    double m_cur_stamp; 
    bool m_bReady;
    r_iter m_gtIter; 
    r_iter m_visIter; 
    r_iter m_dptIter;
};

#endif
