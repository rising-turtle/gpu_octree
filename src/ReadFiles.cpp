#include "ReadFiles.h"
#include <sstream>
#include <iostream>
#include <fstream>
#include <cmath>

CReadFiles::CReadFiles(string dir): 
m_dir(dir),
m_bReady(false)
{
    initDir(m_dir);
}
CReadFiles::~CReadFiles(){}

void CReadFiles::initDir(string dir)
{
    if(m_bReady)
    {
        clearRecords();
    }
    m_dir = dir;
    stringstream str1, str2, str3; 
    str1<<dir<<"/groundtruth.txt";
    str2<<dir<<"/rgb.txt";
    str3<<dir<<"/depth.txt";
    if(!readFiles(str1.str(), m_gtRecord))
    {   
        cout<<"ReadFiles.cpp fail to read file: "<<str1.str()<<" !"<<endl;
        clearRecords();
        return; 
    }
    if(!readFiles(str2.str(), m_visRecord))
    {
        cout<<"ReadFiles.cpp failed to read file: "<<str2.str()<<" !"<<endl;
        clearRecords();
        return ;
    }
    if(!readFiles(str3.str(), m_dptRecord))
    {
        cout<<"ReadFiles.cpp failed to read file: "<<str3.str()<<" !"<<endl;
        clearRecords();
        return ;
    }
    m_gtIter = m_gtRecord.begin();
    m_visIter = m_visRecord.begin();
    m_dptIter = m_dptRecord.begin();
    // cout<<"ReadFiles.cpp: gt.size: "<<m_gtRecord.size()<<" vis.size: "<<m_visRecord.size()<<" dpt.size: "<<m_dptRecord.size()<<endl;
    m_bReady = true;
}

bool CReadFiles::getCurGtStr(double & timestamp, string& gt_v)
{
    if(m_gtIter == m_gtRecord.end())
    {
        cerr<<"ReadFiles.cpp: m_gtIter not valid!"<<endl;
        return false;
    }
    timestamp = m_gtIter->first; 
    gt_v = m_gtIter->second;
}

void CReadFiles::clearRecords()
{
    {
        Record tmp;
        m_gtRecord.swap(tmp);
        m_gtIter = m_gtRecord.begin();
    }
    {
        Record tmp;
        m_visRecord.swap(tmp);
        m_visIter = m_visRecord.begin();
    }
    {
        Record tmp;
        m_dptRecord.swap(tmp);
        m_dptIter = m_dptRecord.begin();
    }
    m_dir.clear();
    m_bReady = false;
}

bool CReadFiles::readFiles(string f, Record& record)
{
    ifstream inf(f.c_str());
    if(inf)
    {
        char tmp[256];
        while(inf.getline(tmp,256))
        {
            if(tmp[0] == '#') continue;
            Key k; 
            Value v; 
            sscanf(tmp, "%lf", &k);
            string t(tmp);
            v = t.substr(t.find(' ')+1, t.npos);
            record.insert(make_pair<Key, Value>(k,v));
        }
    }else 
        return false;
    return true;
}

void CReadFiles::test()
{
    stringstream ss1, ss2, ss3;
    ss1<<m_dir<<"/gt_test.txt";
    ss2<<m_dir<<"/rgb_test.txt";
    ss3<<m_dir<<"/dpt_test.txt";
    
    Record m_gtRecord2, m_visRecord2, m_dptRecord2;
    string s1, s2, s3;
    while(1)
    {
        r_iter it = m_dptIter;
        if(it == m_dptRecord.end()) break;
        m_dptRecord2.insert(make_pair<Key, Value>(it->first,it->second));
        if(!readRecord(s1,s2,s3)) break;
        m_visRecord2.insert(make_pair<Key, Value>(m_visIter->first, m_visIter->second));
        m_gtRecord2.insert(make_pair<Key, Value>(m_gtIter->first, m_gtIter->second));
    }
    // cout<<"ReadFiles.cpp: gt.size: "<<m_gtRecord2.size()<<" vis.size: "<<m_visRecord2.size()<<" dpt.size: "<<m_dptRecord2.size()<<endl;

    dumpFiles(ss1.str(), m_gtRecord2);
    dumpFiles(ss2.str(), m_visRecord2);
    dumpFiles(ss3.str(), m_dptRecord2);
}

void CReadFiles::dumpFiles(string f, Record& r)
{
    ofstream ouf(f.c_str());
    ouf.precision(6);
    r_iter it = r.begin();
    while(it!=r.end())
    {
        ouf<<fixed<<it->first<<" "<<it->second<<endl;
        it++;
    }
}

// get records according to depth_timestamp
bool CReadFiles::readRecord(string & vis_f, string& dth_f, string& gt_v)
{
    string tvis_f, tdth_f, tgt_v;
    if(!m_bReady)
    {
        cout<<"ReadFiles.cpp fail to read files!"<<endl;
        return false;
    }
    if(m_dptIter == m_dptRecord.end()) return false;
    tdth_f = m_dptIter->second;
    if(!getCloseValue(m_visIter, m_visRecord.end(), m_dptIter->first, tvis_f)) 
        return false;
    if(!getCloseValue(m_gtIter, m_gtRecord.end(), m_dptIter->first, tgt_v))
        return false;
    m_cur_stamp = m_dptIter->first;
    ++m_dptIter;
    dth_f = m_dir + string("/") + tdth_f;
    vis_f = m_dir + string("/") + tvis_f;
    gt_v = m_dir + string("/") + tgt_v;

    return true;
}

bool CReadFiles::getCloseValue(CReadFiles::r_iter& it, CReadFiles::r_iter last, Key key, Value& v)
{
    if(it->first >= key) 
    {
        cout<<"ReadFiles.cpp: not moving index with value: "<<it->first<<" key: "<<key<<endl;
        v = it->second;
        return true;
    }
    r_iter it_next = it ;
    ++it_next;
    while(it_next->first < key)
    {
        it++;
        it_next++;
        if(it_next == last) return false;
    }
    if(fabs(key - it->first) > fabs(it_next->first - key))
    {
        it = it_next;
    }
    v = it->second;
}


