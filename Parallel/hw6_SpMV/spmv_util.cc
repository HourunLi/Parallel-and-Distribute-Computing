
#include "spmv_util.h"


bool LoadSourceFromFile(
    const char* filename,
    char* & sourceCode )
{
    bool error = false;
    FILE* fp = NULL;
    int nsize = 0;

    // Open the shader file

    fp = fopen(filename, "rb");
    if( !fp )
    {
        error = true;
    }
    else
    {
        // Allocate a buffer for the file contents
        fseek( fp, 0, SEEK_END );
        nsize = ftell( fp );
        fseek( fp, 0, SEEK_SET );

        sourceCode = new char [ nsize + 1 ];
        if( sourceCode )
        {
            fread( sourceCode, 1, nsize, fp );
            sourceCode[ nsize ] = 0; // Don't forget the NULL terminator
        }
        else
        {
            error = true;
        }

        fclose( fp );
    }

    return error;
}





/*
int findPaddedSize(int realSize, int alignment)
{
    if (realSize % alignment == 0)
	return realSize;
    return realSize + alignment - realSize % alignment;
}

double distance(float* vec1, float* vec2, int size)
{
	double sum = 0.0f;
	for (int i = 0; i < size; i++)
	{
		double tmp = vec1[i] - vec2[i];
		sum += tmp * tmp;
	}
	return sqrt(sum);
}
*/


/*
void two_vec_compare(float* coovec, float* newvec, int size)
{
    double dist = distance(coovec, newvec, size);

    double maxdiff = 0.0f;
    int maxdiffid = 0;
    double maxratiodiff = 0.0f;
    int count = 0;
    for (int i = 0; i < size; i++)
    {
	float tmpa = coovec[i];
	if (tmpa < 0)
	    tmpa *= (-1);
	float tmpb = newvec[i];
	if (tmpb < 0)
	    tmpb *= (-1);
	double diff = tmpa - tmpb;
	if (diff < 0)
	    diff *= (-1);
	float maxab = (tmpa > tmpb)?tmpa:tmpb;
	double ratio = 0.0f;
	if (maxab > 0)
	    ratio = diff / maxab;
	if (diff > maxdiff)
	{
	    maxdiff = diff;
	    maxdiffid = i;
	}
	if (ratio > maxratiodiff)
	    maxratiodiff = ratio;
	if (coovec[i] != newvec[i] && count < 10)
	{
	    printf("Error i %d coo res %f res %f \n", i, coovec[i], newvec[i]);
	    count++;
	}
    }
    printf("Max diff id %d coo res %f res %f \n", maxdiffid, coovec[maxdiffid], newvec[maxdiffid]);
    printf("\nCorrectness Check: Distance %e max diff %e max diff ratio %e vec size %d\n", dist, maxdiff, maxratiodiff, size);
}
*/

/*
void two_vec_compare(int* coovec, int* newvec, int size)
{
//    double dist = distance(coovec, newvec, size);

    double maxdiff = 0.0f;
    int maxdiffid = 0;
    double maxratiodiff = 0.0f;
    int count = 0;
    for (int i = 0; i < size; i++)
    {
	int tmpa = coovec[i];
	if (tmpa < 0)
	    tmpa *= (-1);
	int tmpb = newvec[i];
	if (tmpb < 0)
	    tmpb *= (-1);
	double diff = tmpa - tmpb;
	if (diff < 0)
	    diff *= (-1);
	float maxab = (tmpa > tmpb)?tmpa:tmpb;
	double ratio = 0.0f;
	if (maxab > 0)
	    ratio = diff / maxab;
	if (diff > maxdiff)
	{
	    maxdiff = diff;
	    maxdiffid = i;
	}
	if (ratio > maxratiodiff)
	    maxratiodiff = ratio;
	if (coovec[i] != newvec[i] && count < 10)
	{
	    printf("Error i %d coo res %d res %d \n", i, coovec[i], newvec[i]);
	    count++;
	}
    }
    printf("Max diff id %d coo res %d res %d \n", maxdiffid, coovec[maxdiffid], newvec[maxdiffid]);
    printf("\nCorrectness Check: Distance N max diff %e max diff ratio %e vec size %d\n",  maxdiff, maxratiodiff, size);
}
*/


double timestamp ()
{
    struct timeval tv;
    gettimeofday (&tv, 0);
    return tv.tv_sec + 1e-6*tv.tv_usec;
}






//Read from matrix market format to a coo mat
//template <class dimType, class dataType>
void ReadMMF(char* filename, coo_matrix<int, float>* mat)
{
    FILE* infile = fopen(filename, "r");
    char tmpstr[100];
    char tmpline[1030];
    fscanf(infile, "%s", tmpstr);
    fscanf(infile, "%s", tmpstr);
    fscanf(infile, "%s", tmpstr);
    fscanf(infile, "%s", tmpstr);
    bool ifreal = false;
    if (strcmp(tmpstr, "real") == 0)
        ifreal = true;
    bool ifsym = false;
    fscanf(infile, "%s", tmpstr);
    if (strcmp(tmpstr, "symmetric") == 0)
        ifsym = true;
    int height = 0;
    int width = 0;
    int nnz = 0;
    while (true)
    {
        fscanf(infile, "%s", tmpstr);
        if (tmpstr[0] != '%')
        {
            height = atoi(tmpstr);
            break;
        }
        fgets(tmpline, 1025, infile);
    }
    
    fscanf(infile, "%d %d", &width, &nnz);
    mat->matinfo.height = height;
    mat->matinfo.width = width;
    int* rows = (int*)malloc(sizeof(int)*nnz);
    int* cols = (int*)malloc(sizeof(int)*nnz);
    float* data = (float*)malloc(sizeof(float)*nnz);
    int diaCount = 0;
    for (int i = 0; i < nnz; i++)
    {
        int rowid = 0;
        int colid = 0;
        fscanf(infile, "%d %d", &rowid, &colid);
        rows[i] = rowid - 1;
        cols[i] = colid - 1;
        data[i] = 1.0f;
        if (ifreal)
        {
            double dbldata = 0.0f;
            fscanf(infile, "%lf", &dbldata);
            data[i] = (float)dbldata;
        }
        if (rows[i] == cols[i])
            diaCount++;
    }
    
    if (ifsym)
    {
        int newnnz = nnz * 2 - diaCount;
        mat->matinfo.nnz = newnnz;
        mat->coo_row_id = (int*)malloc(sizeof(int)*newnnz);
        mat->coo_col_id = (int*)malloc(sizeof(int)*newnnz);
        mat->coo_data = (float*)malloc(sizeof(float)*newnnz);
        int matid = 0;
        for (int i = 0; i < nnz; i++)
        {
            mat->coo_row_id[matid] = rows[i];
            mat->coo_col_id[matid] = cols[i];
            mat->coo_data[matid] = data[i];
            matid++;
            if (rows[i] != cols[i])
            {
                mat->coo_row_id[matid] = cols[i];
                mat->coo_col_id[matid] = rows[i];
                mat->coo_data[matid] = data[i];
                matid++;
            }
        }
        assert(matid == newnnz);
        bool tmp = sort_coo<int, float>(mat);
        assert(tmp == true);
    }
    else
    {
        mat->matinfo.nnz = nnz;
        mat->coo_row_id = (int*)malloc(sizeof(int)*nnz);
        mat->coo_col_id = (int*)malloc(sizeof(int)*nnz);
        mat->coo_data = (float*)malloc(sizeof(float)*nnz);
        memcpy(mat->coo_row_id, rows, sizeof(int)*nnz);
        memcpy(mat->coo_col_id, cols, sizeof(int)*nnz);
        memcpy(mat->coo_data, data, sizeof(float)*nnz);
        if (!if_sorted_coo<int, float>(mat))
            sort_coo<int, float>(mat);
        //assert(if_sorted_coo(mat) == true);
    }
    
    fclose(infile);
    free(rows);
    free(cols);
    free(data);
}


/*
void printMatInfo(coo_matrix<int, float>* mat)
{
    printf("\nMatInfo: Width %d Height %d NNZ %d\n", mat->matinfo.width, mat->matinfo.height, mat->matinfo.nnz);
    int minoffset = mat->matinfo.width;
    int maxoffset = -minoffset;
    int nnz = mat->matinfo.nnz;
    int lessn16 = 0;
    int inn16 = 0;
    int less16 = 0;
    int large16 = 0;
    for (int i = 0; i < nnz; i++)
    {
	int rowid = mat->coo_row_id[i];
	int colid = mat->coo_col_id[i];
	int diff = rowid - colid;
	if (diff < minoffset)
	    minoffset = diff;
	if (diff > maxoffset)
	    maxoffset = diff;
	if (diff < -15)
	    lessn16++;
	else if (diff < 0)
	    inn16++;
	else if (diff < 16)
	    less16++;
	else
	    large16++;
    }
    printf("Max Offset %d Min Offset %d\n", maxoffset, minoffset);
    printf("Histogram: <-15: %d -15~-1 %d < 0-15 %d > 16 %d\n", lessn16, inn16, less16, large16);

    if (!if_sorted_coo(mat))
    {
	assert(sort_coo(mat) == true);
    }

    int* cacheperrow = (int*)malloc(sizeof(int)*mat->matinfo.height);
    int* elemperrow = (int*)malloc(sizeof(int)*mat->matinfo.height);
    memset(cacheperrow, 0, sizeof(int)*mat->matinfo.height);
    memset(elemperrow, 0, sizeof(int)*mat->matinfo.height);
    int index = 0;
    for (int i = 0; i < mat->matinfo.height; i++)
    {
	if (i < mat->coo_row_id[index])
	    continue;
	int firstline = mat->coo_col_id[index]/16;
	cacheperrow[i] = 1;
	elemperrow[i] = 1;
	index++;
	while (mat->coo_row_id[index] == i)
	{
	    int nextline = mat->coo_col_id[index]/16;
	    if (nextline != firstline)
	    {
		firstline = nextline;
		cacheperrow[i]++;
	    }
	    elemperrow[i]++;
	    index++;
	}
    }
    int maxcacheline = 0;
    int mincacheline = 100000000;
    int sum = 0;
    for (int i = 0; i < mat->matinfo.height; i++)
    {
	if (cacheperrow[i] < mincacheline)
	    mincacheline = cacheperrow[i];
	if (cacheperrow[i] > maxcacheline)
	    maxcacheline = cacheperrow[i];
	sum += cacheperrow[i];
    }
    printf("Cacheline usage per row: max %d min %d avg %f\n", maxcacheline, mincacheline, (double)sum/(double)mat->matinfo.height);
}
*/





template <class dimType, class dataType>
void coo_spmv(coo_matrix<dimType, dataType>* mat, dataType* vec, dataType* result, dimType vec_size)
{
    //for (dimType i = (dimType)0; i < mat->matinfo.height; i++)
	//result[i] = (dataType)0;
    dimType nnz = mat->matinfo.nnz;
    for (dimType i = (dimType)0; i < nnz; i++)
    {
	dimType row = mat->coo_row_id[i];
	dimType col = mat->coo_col_id[i];
	dataType data = mat->coo_data[i];
	result[row] += data * vec[col];
    }
}


/*
void spmv_only(coo_matrix<int, float>* mat, float* vec, float* coores)
{
    int ressize = mat->matinfo.height;
    for (int i = 0; i < ressize; i++)
	coores[i] = (float)0;
    coo_spmv<int, float>(mat, vec, coores, mat->matinfo.width);
}
*/

/*
void pad_csr(csr_matrix<int, float>* source, csr_matrix<int, float>* dest, int alignment)
{
	using namespace std;	
	dest->matinfo.height = source->matinfo.height;
	dest->matinfo.width = source->matinfo.width;
	dest->csr_row_ptr = (int*)malloc(sizeof(int)*(source->matinfo.height+1));
	vector<int> padcol;
	vector<float> paddata;
	padcol.reserve(source->matinfo.nnz*2);
	paddata.reserve(source->matinfo.nnz*2);
	
	dest->csr_row_ptr[0] = 0;
	
	for (int row = 0; row < source->matinfo.height; row++)
	{
		int start = source->csr_row_ptr[row];
		int end = source->csr_row_ptr[row+1];
		int size = end - start;
		int paddedsize = findPaddedSize(size, alignment);
		dest->csr_row_ptr[row+1] = dest->csr_row_ptr[row] + paddedsize;
		int i = 0;
		for (; i < size; i++)
		{
			padcol.push_back(source->csr_col_id[start + i]);
			paddata.push_back(source->csr_data[start + i]);
		}
		int lastcol = padcol[padcol.size() - 1];
		for (; i < paddedsize; i++)
		{
			padcol.push_back(lastcol);
			paddata.push_back(0.0f);
		}
	}
	dest->csr_col_id = (int*)malloc(sizeof(int)*padcol.size());
	dest->csr_data = (float*)malloc(sizeof(float)*paddata.size());
	dest->matinfo.nnz = padcol.size();
	for (unsigned int i = 0; i < padcol.size(); i++)
	{
		dest->csr_col_id[i] = padcol[i];
		dest->csr_data[i] = paddata[i];
	}
}
*/
